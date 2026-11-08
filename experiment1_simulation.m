function folder = experiment1_simulation
    root = fileparts(mfilename('fullpath'));
    maxNumCompThreads(1);
    folder = fullfile(root, 'results_simulation');
    mkdir(folder);
    [simulationData, simulationInputs] = generate_simulation_data;
    recordIds = 1:2400;
    resultRows = struct([]);
    plotExamples = cell(2, 1);
    for recordId = recordIds(:).'
        record = simulationData.Records(recordId);
        result = identify_system(simulationInputs{record.CaseID}, record.Y, 'simulation');
        scored = score_results(result, record.Indices);
        for scoreIndex = 1:numel(scored)
            scored(scoreIndex).RecordID = recordId;
            scored(scoreIndex).CaseID = record.CaseID;
            scored(scoreIndex).Trial = record.Trial;
            scored(scoreIndex).Count = record.Count;
            scored(scoreIndex).Multiplier = record.Multiplier;
        end
        resultRows = [resultRows, scored];
        if ismember(recordId, [1 1201])
            plotExamples{record.CaseID} = struct('Case', simulationData.Cases{record.CaseID}, 'Record', ...
                record, 'Detected', result.Proposed.FinalCandidates);
        end
        fprintf('Simulation %d / 2400: %s\n', recordId, result.Proposed.Status);
    end
    resultTable = struct2table(resultRows);
    resultTable.FP = resultTable.OutsideAlerts;
    writetable(resultTable, fullfile(folder, 'per_run.csv'));
    invalid = ~resultTable.NormalStop | ~resultTable.ClassificationAvailable;
    resultTable{invalid, {'TP', 'FP', 'FN'}} = NaN;
    [group, means] = findgroups(resultTable(:, {'Method', 'Phase', 'CaseID', 'Count', 'Multiplier'}));
    means.Runs = splitapply(@numel, resultTable.TP, group);
    means.MeanTP = splitapply(@(x) mean(x, 'includemissing'), resultTable.TP, group);
    means.MeanFP = splitapply(@(x) mean(x, 'includemissing'), resultTable.FP, group);
    means.MeanFN = splitapply(@(x) mean(x, 'includemissing'), resultTable.FN, group);
    writetable(means, fullfile(folder, 'condition_means.csv'));
    plot_simulation(folder, plotExamples);
    fprintf('Results: %s (%d of 2400 records run)\n', folder, numel(recordIds));
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

function plot_simulation(folder, examples)
    figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1000 650]);
    titles = {'(a) Multilevel input', '(b) Two-sine input'};
    for inputIndex = 1:2
        inputCase = examples{inputIndex}.Case;
        record = examples{inputIndex}.Record;
        detected = examples{inputIndex}.Detected;
        subplot(2, 1, inputIndex);
        hold on;
        plot(inputCase.Time, inputCase.CleanY, 'k-', 'DisplayName', 'Clean output');
        plot(inputCase.Time, record.Y, '.', 'Color', [.65 .65 .65], 'MarkerSize', 3, 'DisplayName', 'Observed');
        plot(inputCase.Time(record.Indices), record.Y(record.Indices), 'o', 'Color', [.9 .4 .05], 'MarkerSize', ...
            5, 'DisplayName', 'Injected');
        missed = setdiff(record.Indices, detected);
        missedTimes = [NaN; inputCase.Time(missed)];
        missedValues = [NaN; record.Y(missed)];
        plot(missedTimes, missedValues, 'rx', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Missed anomaly');
        xlabel('Time (s)');
        ylabel('Output');
        title(titles{inputIndex});
        legend('Location', 'northoutside', 'Orientation', 'horizontal');
        box on;
    end
    exportgraphics(figureHandle, fullfile(folder, 'simulation.png'), 'Resolution', 300);
    close(figureHandle);
end
