function folder = experiment3_damadics
    root = fileparts(mfilename('fullpath'));
    maxNumCompThreads(1);
    folder = fullfile(root, 'results_damadics');
    mkdir(folder);
    file = fullfile(root, '17112001.txt');
    rawData = readmatrix(file, 'FileType', 'text');
    test = rawData(58001:60001,:);
    rawtime = test(:, 1);
    sampleTimes = (rawtime - rawtime(1)) / 10;
    inputValues = (100 - test(:, 28)) / 45.5;
    normalizedOutput = test(:, 27) / 2.9622624731782294;
    integrationPlan = struct('Dt', sampleTimes(2) - sampleTimes(1), 'Substeps', 64, 'Values', repmat(inputValues(1:end - 1).', ...
        65, 1), ...
        'Time', sampleTimes, 'Input', struct('Kind', 'piecewise'), 'N', numel(normalizedOutput));
    inputData = struct('Order', 2, 'NumeratorOrder', 0, 'Time', sampleTimes, 'InitialState', [median(normalizedOutput(1:5)); 0], ...
        'Plan', integrationPlan);
    anomalyIndices = find(rawtime >= 58150 & rawtime < 58325);
    result = identify_system(inputData, normalizedOutput, 'damadics');
    resultTable = struct2table(score_results(result, anomalyIndices));
    writetable(resultTable, fullfile(folder, 'detection.csv'));
    proposedResult = result.Proposed;
    if proposedResult.ClassificationAvailable
        detected = proposedResult.FinalCandidates;
        detectedFaultIndices = intersect(anomalyIndices, detected);
        missedFaultIndices = setdiff(anomalyIndices, detected);
        outsideAlertIndices = setdiff(detected, anomalyIndices);
        figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1050 600]);
        subplot(2, 1, 1);
        plot(rawtime, test(:, 27), 'Color', [.5 .5 .5], 'DisplayName', 'Measured flow');
        hold on;
        plot(rawtime(detectedFaultIndices), test(detectedFaultIndices, 27), 'o', 'Color', [.9 .4 .05], ...
            'MarkerSize', 4, 'DisplayName', 'Detected fault');
        plot(rawtime(missedFaultIndices), test(missedFaultIndices, 27), 'rx', 'LineWidth', 1.3, 'DisplayName', ...
            'Missed fault');
        plot(rawtime(outsideAlertIndices), test(outsideAlertIndices, 27), 's', 'Color', [.5 .15 .6], 'DisplayName', ...
            'Outside alert');
        ylabel('Flow (t/h)');
        xlabel('Original time (s)');
        legend('Location', 'northoutside', 'Orientation', 'horizontal');
        box on;
        subplot(2, 1, 2);
        plot(rawtime, ismember((1:numel(normalizedOutput)).', anomalyIndices), 'k-', 'DisplayName', 'Annotated interval');
        hold on;
        plot(rawtime(detected), ones(numel(detected), 1), 'o', 'Color', [.9 .4 .05], 'MarkerSize', 4, ...
            'DisplayName', 'Detected');
        xlabel('Original time (s)');
        ylabel('Fault label / detection');
        ylim([-.1 1.25]);
        legend('Location', 'best');
        box on;
        exportgraphics(figureHandle, fullfile(folder, 'damadics.png'), 'Resolution', 300);
        close(figureHandle);
    end
    disp(resultTable(:, {'Method', 'Phase', 'Status', 'NormalStop', 'TP', 'FN', 'OutsideAlerts', 'Recall'}));
    fprintf('Outside-interval alerts have unknown truth. Results: %s\n', folder);
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

