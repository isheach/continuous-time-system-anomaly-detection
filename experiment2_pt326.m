function folder = experiment2_pt326
    seeds = [1919514327 3468051581 3574056783 3711408868 981187083];
    root = fileparts(mfilename('fullpath'));
    maxNumCompThreads(1);
    folder = fullfile(root, 'results_pt326');
    mkdir(folder);
    file = fullfile(root, 'dryer2.mat');
    rawData = load(file, 'u2', 'y2');
    calibration = struct('SigmaPhysical', .15617366985887576, 'YOffset', 4.793338793333344, ...
        'YScale', .8356225886722599, 'UOffset', 4.99, 'UScale', 1.4978651474682227);
    inputValues = (rawData.u2(:) - calibration.UOffset) / calibration.UScale;
    firstSample = 401;
    lastSample = 1000;
    sampleCount = 600;
    rawtime = (0:999).' * .08;
    inputData = struct('Order', 2, 'NumeratorOrder', 0, 'Time', (rawtime(firstSample:lastSample) - rawtime(firstSample)) / .08, ...
        'U', inputValues(firstSample:lastSample), 'DelaySamples', 2, 'InputPrehistory', inputValues(399:400));
    recordedOutput = rawData.y2(firstSample:lastSample);
    recordedOutput = recordedOutput(:);
    injections = pcg64_injections(seeds, sampleCount, 12, calibration.SigmaPhysical, folder);
    resultRows = struct([]);
    diagnosis = struct('Group', {}, 'Record', {}, 'OriginalIndex', {}, ...
        'BackgroundResidual', {}, 'Injection', {}, 'Center', {}, 'CenteredResidual', {}, 'Threshold', ...
            {});
    for copyIndex = 0:8 * numel(seeds)
        if copyIndex == 0
            groupIndex = 0;
            recordIndex = 0;
            baseSeed = 0;
            derivedSeed = 0;
            anomalyType = "Control";
            amplitudeMultiplier = 0;
            repeatIndex = 0;
            anomalyIndices = [];
            delta = zeros(sampleCount, 1);
        else
            groupIndex = ceil(copyIndex / 8);
            recordIndex = mod(copyIndex - 1, 8) + 1;
            injectionRows = injections(injections.Group == groupIndex & injections.Record == recordIndex, ...
                :);
            baseSeed = injectionRows.BaseSeed(1);
            derivedSeed = injectionRows.DerivedSeed(1);
            anomalyType = string(injectionRows.Type(1));
            amplitudeMultiplier = injectionRows.Amplitude(1);
            repeatIndex = injectionRows.Repeat(1);
            anomalyIndices = injectionRows.Index;
            delta = zeros(sampleCount, 1);
            delta(anomalyIndices) = injectionRows.AddedPhysicalV;
        end
        contaminatedOutput = recordedOutput + delta;
        normalizedOutput = (contaminatedOutput - calibration.YOffset) / calibration.YScale;
        result = identify_system(inputData, normalizedOutput, 'pt326');
        scored = score_results(result, anomalyIndices);
        for scoreIndex = 1:numel(scored)
            scored(scoreIndex).Group = groupIndex;
            scored(scoreIndex).BaseSeed = baseSeed;
            scored(scoreIndex).DerivedSeed = derivedSeed;
            scored(scoreIndex).Record = recordIndex;
            scored(scoreIndex).Type = anomalyType;
            scored(scoreIndex).Amplitude = amplitudeMultiplier;
            scored(scoreIndex).Repeat = repeatIndex;
            scored(scoreIndex).InjectionIndices = join(string(anomalyIndices.'), ';');
            scored(scoreIndex).InjectionOriginalIndices = join(string(anomalyIndices.' + 400), ';');
            scored(scoreIndex).AddedPhysicalV = join(compose('%.17g', delta(anomalyIndices).'), ';');
            detectedIndices = result.(scored(scoreIndex).Method).FinalCandidates;
            if scored(scoreIndex).Phase == "BeforeReview"
                detectedIndices = result.(scored(scoreIndex).Method).PreReviewCandidates;
            end
            scored(scoreIndex).DetectedOriginalIndices = join(string(detectedIndices.' + 400), ';');
        end
        resultRows = [resultRows, scored];
        if copyIndex > 0 && result.Proposed.ClassificationAvailable
            methodResult = result.Proposed;
            residual = normalizedOutput - methodResult.Prediction;
            center = median(residual);
            threshold = 3 * max(1e-12, 1.4826 * median(abs(residual - center)));
            for missedIndex = setdiff(anomalyIndices, methodResult.FinalCandidates).'
                backgroundResidual = (recordedOutput(missedIndex) - calibration.YOffset) / calibration.YScale - methodResult.Prediction(missedIndex);
                injectionValue = delta(missedIndex) / calibration.YScale;
                diagnosis(end + 1) = struct('Group', groupIndex, 'Record', recordIndex, 'OriginalIndex', ...
                    missedIndex + 400, ...
                    'BackgroundResidual', backgroundResidual, 'Injection', injectionValue, 'Center', center, ...
                    'CenteredResidual', backgroundResidual + injectionValue - center, 'Threshold', threshold);
            end
        end
        if groupIndex == 4 && recordIndex == 2 && result.Proposed.ClassificationAvailable
            plot_pt326(folder, recordedOutput, contaminatedOutput, anomalyIndices, result.Proposed.FinalCandidates, ...
                firstSample);
        end
        fprintf('PT326 group %d, copy %d: %s\n', groupIndex, recordIndex, result.Proposed.Status);
    end
    resultTable = struct2table(resultRows);
    writetable(resultTable(resultTable.Group > 0,:), fullfile(folder, 'per_copy.csv'));
    writetable(resultTable(resultTable.Group == 0,:), fullfile(folder, 'control.csv'));
    writetable(struct2table(diagnosis), fullfile(folder, 'missed_diagnosis.csv'));
    summarize_pt326(resultTable, folder, calibration.SigmaPhysical, numel(seeds));
    fprintf('FP counts alerts without an artificial injection; background normality is not fully labeled.\nResults: %s\n', ...
        folder);
end

function injectionTable = pcg64_injections(seeds, sampleCount, injectionsPerCopy, sigmaPhysical, folder)
    rows = repmat(struct('Group', 0, 'Record', 0, 'BaseSeed', 0, 'DerivedSeed', 0, ...
        'Type', "", 'Amplitude', 0, 'Repeat', 0, 'Index', 0, 'AddedPhysicalV', 0), ...
        numel(seeds) * 8 * injectionsPerCopy, 1);
    typeNames = ["isolated_spike", "short_block3"];
    rowIndex = 0;
    for groupIndex = 1:numel(seeds)
        recordIndex = 0;
        for kind = 1:2
            for amplitude = [2.5 5]
                for repeatIndex = 1:2
                    recordIndex = recordIndex + 1;
                    derivedSeed = seeds(groupIndex) + 10 * kind + repeatIndex;
                    generator = pcg_seed(derivedSeed);
                    occupied = false(sampleCount, 1);
                    eventFactors = zeros(sampleCount, 1);
                    remaining = injectionsPerCopy;
                    while remaining > 0
                        eventLength = min(2 * kind - 1, remaining);
                        [firstIndex, generator] = pcg_position(generator, sampleCount - eventLength + 1);
                        indices = firstIndex:firstIndex + eventLength - 1;
                        neighbors = max(1, firstIndex - 1):min(sampleCount, indices(end) + 1);
                        if any(occupied(neighbors))
                            continue;
                        end
                        [signValue, generator] = pcg_double(generator);
                        [uniformValue, generator] = pcg_double(generator);
                        sign = 2 * (signValue < .5) - 1;
                        jitter = .8 + (1.2 - .8) * uniformValue;
                        occupied(indices) = true;
                        eventFactors(indices) = sign * jitter;
                        remaining = remaining - eventLength;
                    end
                    for index = find(occupied).'
                        rowIndex = rowIndex + 1;
                        rows(rowIndex) = struct('Group', groupIndex, 'Record', recordIndex, ...
                            'BaseSeed', seeds(groupIndex), 'DerivedSeed', derivedSeed, ...
                            'Type', typeNames(kind), 'Amplitude', amplitude, 'Repeat', repeatIndex, ...
                            'Index', index, 'AddedPhysicalV', amplitude * sigmaPhysical * eventFactors(index));
                    end
                end
            end
        end
    end
    injectionTable = struct2table(rows);
    file = fopen(fullfile(folder, 'injections.csv'), 'w', 'n', 'UTF-8');
    fprintf(file, 'Group,Record,BaseSeed,DerivedSeed,Type,Amplitude,Repeat,Index,AddedPhysicalV\n');
    for row = rows.'
        fprintf(file, '%d,%d,%.0f,%.0f,%s,%.1f,%d,%d,%.17g\n', row.Group, row.Record, ...
            row.BaseSeed, row.DerivedSeed, row.Type, row.Amplitude, row.Repeat, row.Index, row.AddedPhysicalV);
    end
    fclose(file);
end

function generator = pcg_seed(seed)
    entropy = uint32([mod(seed, 4294967296), floor(seed / 4294967296), 0, 0]);
    pool = zeros(1, 4, 'uint32');
    hashConstant = uint32(hex2dec('43b0d7e5'));
    for index = 1:4
        [pool(index), hashConstant] = pcg_hash(entropy(index), hashConstant);
    end
    for source = 1:4
        for destination = 1:4
            if source ~= destination
                [hashed, hashConstant] = pcg_hash(pool(source), hashConstant);
                left = multiply32(uint32(hex2dec('ca01f9dd')), pool(destination));
                right = multiply32(uint32(hex2dec('4973f715')), hashed);
                mixed = uint32(mod(double(left) - double(right), 4294967296));
                pool(destination) = bitxor(mixed, bitshift(mixed, -16));
            end
        end
    end
    words = zeros(1, 8, 'uint32');
    hashConstant = uint32(hex2dec('8b51f9dd'));
    for index = 1:8
        value = bitxor(pool(mod(index - 1, 4) + 1), hashConstant);
        hashConstant = multiply32(hashConstant, uint32(hex2dec('58f38ded')));
        value = multiply32(value, hashConstant);
        words(index) = bitxor(value, bitshift(value, -16));
    end
    state = java.math.BigInteger('0');
    sequence = java.math.BigInteger('0');
    for index = [2 1 4 3]
        state = state.shiftLeft(32).add(java.math.BigInteger(sprintf('%.0f', double(words(index)))));
        sequence = sequence.shiftLeft(32).add(java.math.BigInteger(sprintf('%.0f', double(words(index + 4)))));
    end
    generator.Mask128 = java.math.BigInteger('ffffffffffffffffffffffffffffffff', 16);
    generator.Mask64 = java.math.BigInteger('ffffffffffffffff', 16);
    generator.Mask32 = java.math.BigInteger('ffffffff', 16);
    generator.Multiplier = java.math.BigInteger('2360ed051fc65da44385df649fccf645', 16);
    generator.Increment = sequence.shiftLeft(1).or(java.math.BigInteger('1')).and(generator.Mask128);
    state = generator.Increment.add(state).and(generator.Mask128);
    generator.State = state.multiply(generator.Multiplier).add(generator.Increment).and(generator.Mask128);
    generator.HasBuffered = false;
    generator.Buffered = uint32(0);
end

function [value, hashConstant] = pcg_hash(value, hashConstant)
    value = bitxor(value, hashConstant);
    hashConstant = multiply32(hashConstant, uint32(hex2dec('931e8875')));
    value = multiply32(value, hashConstant);
    value = bitxor(value, bitshift(value, -16));
end

function value = multiply32(left, right)
    value = uint32(bitand(uint64(left) * uint64(right), uint64(4294967295)));
end

function [word, generator] = pcg_word(generator)
    generator.State = generator.State.multiply(generator.Multiplier).add(generator.Increment).and(generator.Mask128);
    value = generator.State.shiftRight(64).xor(generator.State.and(generator.Mask64));
    rotation = double(generator.State.shiftRight(122).intValue);
    word = value.shiftRight(rotation).or(value.shiftLeft(64 - rotation)).and(generator.Mask64);
end

function [value, generator] = pcg_double(generator)
    [word, generator] = pcg_word(generator);
    value = word.shiftRight(11).doubleValue * 2^-53;
end

function [position, generator] = pcg_position(generator, range)
    threshold = mod(4294967296, range);
    while true
        if generator.HasBuffered
            value = generator.Buffered;
            generator.HasBuffered = false;
        else
            [word, generator] = pcg_word(generator);
            value = uint32(word.and(generator.Mask32).doubleValue);
            generator.Buffered = uint32(word.shiftRight(32).doubleValue);
            generator.HasBuffered = true;
        end
        product = uint64(value) * uint64(range);
        if double(bitand(product, uint64(4294967295))) >= threshold
            position = double(bitshift(product, -32)) + 1;
            return;
        end
    end
end

function scoreRows = score_results(methodResults, anomalyIndices)
    scoreRows = struct([]);
    anomalyIndices = anomalyIndices(:);
    for methodName = ["Proposed", "Global", "Cumulative"]
        methodResult = methodResults.(methodName);
        phases = "Final";
        if methodName == "Cumulative"
            phases = ["BeforeReview", "Final"];
        end
        for phaseName = phases
            detectedIndices = [];
            truePositiveCount = NaN;
            falseNegativeCount = NaN;
            outsideAlertCount = NaN;
            recallRate = NaN;
            if methodResult.ClassificationAvailable
                detectedIndices = methodResult.FinalCandidates(:);
                if phaseName == "BeforeReview"
                    detectedIndices = methodResult.PreReviewCandidates(:);
                end
                truePositiveCount = numel(intersect(anomalyIndices, detectedIndices));
                falseNegativeCount = numel(setdiff(anomalyIndices, detectedIndices));
                outsideAlertCount = numel(setdiff(detectedIndices, anomalyIndices));
                recallRate = truePositiveCount / numel(anomalyIndices);
            end
            scoreRow = struct('Method', methodName, 'Phase', phaseName, 'Status', string(methodResult.Status), ...
                'NormalStop', methodResult.NormalStop, 'ClassificationAvailable', methodResult.ClassificationAvailable, ...
                'InjectedOrKnownPositive', numel(anomalyIndices), 'TP', truePositiveCount, 'FN', falseNegativeCount, ...
                'OutsideAlerts', outsideAlertCount, 'Recall', recallRate, 'DetectedIndices', join(string(detectedIndices.'), ...
                    ';'), ...
                'ReviewPerformed', methodResult.ReviewPerformed, ...
                'Theta', join(compose('%.17g', methodResult.Theta.'), ';'), ...
                'FitTargets', join(string(methodResult.FitTargets.'), ';'), ...
                'InitialState', join(compose('%.17g', methodResult.InitialState.'), ';'));
            scoreRows = [scoreRows, scoreRow];
        end
    end
end

function summarize_pt326(resultTable, folder, sigmaPhysical, groupCount)
    resultTable = resultTable(resultTable.Group > 0,:);
    groupRows = struct([]);
    conditionRows = struct([]);
    keys = unique(resultTable(:, {'Method', 'Phase'}), 'rows', 'stable');
    for methodIndex = 1:height(keys)
        for groupIndex = 1:groupCount
            selectedRows = resultTable(resultTable.Method == keys.Method(methodIndex) & resultTable.Phase == keys.Phase(methodIndex) & resultTable.Group == groupIndex, ...
                :);
            isComplete = height(selectedRows) == 8 && all(selectedRows.NormalStop & selectedRows.ClassificationAvailable);
            summaryRow = struct('Group', groupIndex, 'BaseSeed', selectedRows.BaseSeed(1), 'Method', keys.Method(methodIndex), ...
                'Phase', keys.Phase(methodIndex), ...
                'Complete', isComplete, 'Injected', sum(selectedRows.InjectedOrKnownPositive), 'TP', sum(selectedRows.TP), ...
                    'FN', sum(selectedRows.FN), 'FPStar', sum(selectedRows.OutsideAlerts), ...
                'NormalCopies', sum(selectedRows.NormalStop & selectedRows.ClassificationAvailable), 'ClassifiedCopies', ...
                    sum(selectedRows.ClassificationAvailable), ...
                'ClassifiedTP', sum(selectedRows.TP), 'ClassifiedFN', sum(selectedRows.FN), 'ClassifiedFPStar', ...
                    sum(selectedRows.OutsideAlerts), ...
                'ReviewCopies', sum(selectedRows.ReviewPerformed));
            if ~isComplete
                summaryRow.TP = NaN;
                summaryRow.FN = NaN;
                summaryRow.FPStar = NaN;
            end
            groupRows = [groupRows, summaryRow];
        end
        for anomalyType = ["isolated_spike", "short_block3"]
            for amplitudeMultiplier = [2.5 5]
                selectedRows = resultTable(resultTable.Method == keys.Method(methodIndex) & resultTable.Phase == keys.Phase(methodIndex) & resultTable.Type == anomalyType & resultTable.Amplitude == amplitudeMultiplier, ...
                    :);
                isComplete = height(selectedRows) == 2 * groupCount && all(selectedRows.NormalStop & selectedRows.ClassificationAvailable);
                summaryRow = struct('Method', keys.Method(methodIndex), 'Phase', keys.Phase(methodIndex), ...
                    'Type', anomalyType, 'Amplitude', amplitudeMultiplier, ...
                    'AmplitudeLowerV', .8 * amplitudeMultiplier * sigmaPhysical, 'AmplitudeUpperV', 1.2 * amplitudeMultiplier * sigmaPhysical, ...
                        'Complete', isComplete, ...
                    'InjectedPerGroup', sum(selectedRows.InjectedOrKnownPositive) / groupCount, 'MeanTP', ...
                        sum(selectedRows.TP) / groupCount, 'MeanFN', sum(selectedRows.FN) / groupCount, 'MeanFPStar', ...
                        sum(selectedRows.OutsideAlerts) / groupCount, ...
                    'NormalCopies', sum(selectedRows.NormalStop & selectedRows.ClassificationAvailable), ...
                        'ClassifiedCopies', sum(selectedRows.ClassificationAvailable), ...
                    'ClassifiedMeanTP', sum(selectedRows.TP) / groupCount, 'ClassifiedMeanFN', sum(selectedRows.FN) / groupCount, ...
                        'ClassifiedMeanFPStar', sum(selectedRows.OutsideAlerts) / groupCount);
                if ~isComplete
                    summaryRow.MeanTP = NaN;
                    summaryRow.MeanFN = NaN;
                    summaryRow.MeanFPStar = NaN;
                end
                conditionRows = [conditionRows, summaryRow];
            end
        end
    end
    groupTable = struct2table(groupRows);
    conditionTable = struct2table(conditionRows);
    writetable(groupTable, fullfile(folder, 'by_group.csv'));
    writetable(conditionTable, fullfile(folder, 'condition_means.csv'));
    proposedTable = sortrows(groupTable(groupTable.Method == "Proposed" & groupTable.Phase == "Final", ...
        :), 'Group');
    summary = table([string(proposedTable.BaseSeed); "合计"], [proposedTable.Injected; sum(proposedTable.Injected)], ...
        [proposedTable.TP; sum(proposedTable.TP)], [proposedTable.FN; sum(proposedTable.FN)], [proposedTable.FPStar; sum(proposedTable.FPStar)], ...
        'VariableNames', {'种子', '注入点', '检出', '漏检', '非注入位置告警'});
    file = fullfile(folder, 'seed_summary.csv');
    fileHandle = fopen(file, 'w', 'n', 'UTF-8');
    fwrite(fileHandle, [239 187 191], 'uint8');
    fprintf(fileHandle, '种子,注入点,检出,漏检,非注入位置告警\n');
    for summaryRow = 1:height(summary)
        fprintf(fileHandle, '%s,%.15g,%.15g,%.15g,%.15g\n', summary{summaryRow, 1}, summary{summaryRow, 2:5});
    end
    fclose(fileHandle);
    disp(summary);
    fprintf('Seed summary CSV: %s\n', file);
    statusRows = struct([]);
    for methodName = ["Proposed", "Global", "Cumulative"]
        selectedRows = resultTable(resultTable.Method == methodName & resultTable.Phase == "Final",:);
        summaryRow = struct('Method', methodName, 'Copies', height(selectedRows), 'NormalCopies', sum(selectedRows.NormalStop), ...
            'ClassifiedCopies', sum(selectedRows.ClassificationAvailable), 'StagnatedCopies', sum(selectedRows.Status == "stagnated"), ...
            'CycleCopies', sum(selectedRows.Status == "mask_cycle" | selectedRows.Status == "iteration_state_cycle"), ...
            'ReviewCopies', sum(selectedRows.ReviewPerformed), 'ClassifiedTP', sum(selectedRows.TP), 'ClassifiedFN', ...
                sum(selectedRows.FN), ...
            'ClassifiedFPStar', sum(selectedRows.OutsideAlerts));
        statusRows = [statusRows, summaryRow];
    end
    writetable(struct2table(statusRows), fullfile(folder, 'method_status.csv'));
end

function plot_pt326(folder, recordedOutput, contaminatedOutput, anomalyIndices, detected, first)
    sampleTimes = (first - 1:first + numel(recordedOutput) - 2).' * 0.08;
    detectedInjectionIndices = intersect(anomalyIndices, detected);
    missedInjectionIndices = setdiff(anomalyIndices, detected);
    outsideAlertIndices = setdiff(detected, anomalyIndices);
    figureHandle = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 6.8 3.7]);
    hold on;
    plot(sampleTimes, recordedOutput, 'k-', 'LineWidth', 1, 'DisplayName', 'Recorded output');
    plot(sampleTimes, contaminatedOutput, '.', 'Color', [.65 .65 .65], 'MarkerSize', 4, 'DisplayName', ...
        'Injected output');
    plot(sampleTimes(detectedInjectionIndices), contaminatedOutput(detectedInjectionIndices), 'o', 'Color', ...
        [.85 .4 .05], 'MarkerSize', 8, 'LineWidth', 1.3, 'DisplayName', 'Detected injection (TP)');
    alertTime = [NaN; sampleTimes(outsideAlertIndices)];
    alertValue = [NaN; contaminatedOutput(outsideAlertIndices)];
    plot(alertTime, alertValue, 's', 'Color', [.5 .15 .6], 'MarkerFaceColor', [.5 .15 .6], 'MarkerSize', ...
        8, 'DisplayName', 'Non-injected alert (FP)');
    missTime = [NaN; sampleTimes(missedInjectionIndices)];
    missValue = [NaN; contaminatedOutput(missedInjectionIndices)];
    plot(missTime, missValue, 'rx', 'MarkerSize', 9, 'LineWidth', 1.5, 'DisplayName', 'Missed injection (FN)');
    set(gca, 'FontSize', 14);
    xlim([sampleTimes(1) sampleTimes(end)]);
    xlabel('Original time (s)', 'FontSize', 15);
    ylabel('Output (V)', 'FontSize', 15);
    legend('Location', 'northoutside', 'NumColumns', 2, 'FontSize', 13, 'Box', 'off');
    box on;
    exportgraphics(figureHandle, fullfile(folder, 'pt326.png'), 'Resolution', 300);
    close(figureHandle);
end
