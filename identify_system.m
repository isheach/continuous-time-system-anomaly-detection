function methodResults = identify_system(inputData, outputValues, experiment)
    settings = configuration(experiment);
    methodResults = struct;
    for methodName = ["Proposed", "Global", "Cumulative"]
        methodResults.(methodName) = detect(inputData, outputValues, settings, methodName);
    end
end

function methodResult = detect(inputData, outputValues, settings, methodName)
    sampleCount = numel(outputValues);
    allIndices = (1:sampleCount).';
    parameterCount = inputData.Order + inputData.NumeratorOrder + 1;
    minimum = max(settings.MinimumTrusted, 2 * parameterCount);
    fitMask = true(sampleCount, 1);
    previousGlobalMask = [];
    previousParameters = [];
    previousStates = {};
    iterationTrace = {};
    lastFit = [];
    lastGlobalMask = [];
    terminationStatus = 'max_iterations';
    for iterationIndex = 0:settings.MaxIterations
        if strcmp(methodName, 'Proposed')
            if nnz(fitMask) < minimum
                terminationStatus = 'insufficient_trusted';
                break;
            end
            if nnz(~fitMask) / sampleCount > settings.MaxMaskedFraction
                terminationStatus = 'mask_fraction_limit';
                break;
            end
        end
        previousStates{end + 1} = struct('I', fitMask, 'PreviousG', previousGlobalMask, 'Warm', previousParameters);
        masked = outputValues;
        masked(~fitMask) = settings.MaskedSentinel;
        if settings.ProfileInitialState
            fitResult = fit_profiled_initial_state(inputData, masked, fitMask, settings, previousParameters);
        else
            fitResult = fit_fixed_initial_state(inputData, masked, fitMask, settings, previousParameters);
        end
        iterationRecord = struct('Iteration', iterationIndex, 'FitTargets', find(fitMask), 'Theta', fitResult.Theta, ...
            'RMSE', fitResult.RMSE, 'GlobalCandidates', [], 'LocalRemoved', [], ...
            'GlobalStable', false, 'Status', 'fitted');
        if ~fitResult.Success
            terminationStatus = fitResult.Failure;
            iterationRecord.Status = terminationStatus;
            iterationTrace{end + 1} = iterationRecord;
            break;
        end
        lastFit = fitResult;
        predictionResiduals = outputValues - fitResult.Prediction;
        nextFitMask = fitMask;
        if strcmp(methodName, 'Cumulative')
            if fitResult.RMSE <= settings.RMSETarget
                terminationStatus = 'rmse_converged';
            elseif iterationIndex == settings.MaxIterations
                terminationStatus = 'max_iterations';
            else
                removed = mad_candidates(predictionResiduals, find(fitMask), find(fitMask), settings.MAD);
                iterationRecord.LocalRemoved = removed;
                nextFitMask(removed) = false;
                if isempty(removed)
                    terminationStatus = 'stagnated';
                elseif nnz(~nextFitMask) / sampleCount > settings.MaxMaskedFraction
                    terminationStatus = 'mask_fraction_limit';
                elseif nnz(nextFitMask) < minimum
                    terminationStatus = 'insufficient_trusted';
                else
                    terminationStatus = 'refit_required';
                end
            end
        else
            rejected = mad_candidates(predictionResiduals, allIndices, allIndices, settings.MAD);
            globalMask = true(sampleCount, 1);
            globalMask(rejected) = false;
            iterationRecord.GlobalCandidates = rejected;
            if strcmp(methodName, 'Global')
                nextFitMask = globalMask;
                if isequal(nextFitMask, fitMask)
                    if fitResult.RMSE <= settings.RMSETarget
                        terminationStatus = 'rmse_converged';
                    else
                        terminationStatus = 'stagnated';
                    end
                elseif nnz(~nextFitMask) / sampleCount > settings.MaxMaskedFraction
                    terminationStatus = 'mask_fraction_limit';
                elseif nnz(nextFitMask) < minimum
                    terminationStatus = 'insufficient_trusted';
                elseif any(cellfun(@(previousState) isequal(previousState.I, nextFitMask), previousStates))
                    terminationStatus = 'mask_cycle';
                elseif iterationIndex == settings.MaxIterations
                    terminationStatus = 'max_iterations';
                else
                    terminationStatus = 'refit_required';
                end
            else
                iterationRecord.GlobalStable = ~isempty(previousGlobalMask) && isequal(globalMask, previousGlobalMask);
                if nnz(~globalMask) / sampleCount > settings.MaxMaskedFraction
                    terminationStatus = 'global_mask_fraction_limit';
                elseif nnz(globalMask) < minimum
                    terminationStatus = 'global_insufficient_trusted';
                else
                    lastGlobalMask = globalMask;
                    if iterationRecord.GlobalStable
                        terminationStatus = 'global_partition_stable';
                    else
                        removed = mad_candidates(predictionResiduals, find(fitMask), find(fitMask), settings.MAD);
                        iterationRecord.LocalRemoved = removed;
                        nextFitMask(removed) = false;
                        if nnz(~nextFitMask) / sampleCount > settings.MaxMaskedFraction
                            terminationStatus = 'local_mask_fraction_limit';
                        elseif nnz(nextFitMask) < minimum
                            terminationStatus = 'local_insufficient_trusted';
                        elseif iterationIndex == settings.MaxIterations
                            terminationStatus = 'max_iterations';
                        elseif any(cellfun(@(previousState) isequal(previousState.I, nextFitMask) && isequal(previousState.PreviousG, ...
                            globalMask) && isequal(previousState.Warm, fitResult.Theta), previousStates))
                            terminationStatus = 'iteration_state_cycle';
                        else
                            terminationStatus = 'refit_required';
                        end
                    end
                end
                previousGlobalMask = globalMask;
            end
        end
        iterationRecord.Status = terminationStatus;
        iterationTrace{end + 1} = iterationRecord;
        if ~strcmp(terminationStatus, 'refit_required')
            break;
        end
        fitMask = nextFitMask;
        previousParameters = fitResult.Theta;
    end
    normalStop = strcmp(terminationStatus, 'global_partition_stable') || strcmp(terminationStatus, 'rmse_converged');
    hasClassification = false;
    parameterVector = nan(parameterCount, 1);
    prediction = nan(sampleCount, 1);
    fittingIndices = [];
    initialState = [];
    if ~isempty(lastFit)
        parameterVector = lastFit.Theta;
        prediction = lastFit.Prediction;
        fittingIndices = lastFit.Targets;
        initialState = lastFit.InitialState;
        hasClassification = lastFit.Success && isequal(fittingIndices, find(fitMask));
    end
    preReviewCandidates = find(~fitMask);
    candidates = preReviewCandidates;
    if strcmp(methodName, 'Proposed')
        hasClassification = ~isempty(lastGlobalMask);
        if hasClassification
            candidates = find(~lastGlobalMask);
        else
            candidates = [];
        end
    end
    reviewPerformed = strcmp(methodName, 'Cumulative') && normalStop;
    if reviewPerformed
        candidates = terminal_review(parameterVector, inputData.Time, outputValues, prediction, preReviewCandidates, ...
            inputData.Order, settings.Review);
    end
    methodResult = struct('Status', terminationStatus, 'NormalStop', normalStop, 'ClassificationAvailable', ...
        hasClassification, ...
        'Theta', parameterVector, 'InitialState', initialState, 'Prediction', prediction, 'FitTargets', ...
            fittingIndices, ...
        'FinalCandidates', candidates, 'PreReviewCandidates', preReviewCandidates, ...
        'ReviewPerformed', reviewPerformed, 'Trace', {iterationTrace});
end

function anomalyIndices = mad_candidates(predictionResiduals, reference, classify, settings)
    residualValues = predictionResiduals(reference);
    residualCenter = median(residualValues);
    residualScale = max(settings.ScaleFloor, 1.4826 * median(abs(residualValues - residualCenter)));
    if max(abs(residualValues)) <= settings.ResidualZeroTolerance
        anomalyIndices = zeros(0, 1);
    else
        anomalyIndices = classify(abs(predictionResiduals(classify) - residualCenter) > settings.SigmaMultiplier * residualScale);
    end
end

function settings = configuration(experiment)
    settings = struct('RMSETarget', 1e-8, 'MaxIterations', 10, 'MaxMaskedFraction', .45, ...
        'MinimumTrusted', 10, 'MaskedSentinel', 1e100, 'WarmFixedStart', 2, 'ProfileInitialState', false);
    settings.MAD = struct('SigmaMultiplier', 3, 'ScaleFloor', 1e-12, 'ResidualZeroTolerance', 1e-8);
    settings.Review = struct('ConditionLimit', 1e8, 'RcondMinimum', 1e-10, 'MinimumReferenceCount', 3, ...
        'Multiplier', 3);
    settings.Fit = struct('Starts', [.2 .5 .2; .6 1.2 .8; 2 3 1.5; 4 1 .5], ...
        'MaxIterations', 350, 'MaxEvaluations', 4500, 'FDStep', .001, ...
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12, 'OptimalityTolerance', 1e-10, ...
        'CoefficientLower', .02, 'CoefficientUpper', 60, 'GainLower', .02, 'GainUpper', 10, ...
        'StabilityMargin', 1e-6, 'JacobianConditionLimit', 1e8);
    switch experiment
        case 'simulation'
            settings.Fit.Starts = [.2 .5 .2 .2; .6 1.2 .5 .8; 2 3 1 1.5; 4 1 .8 .5];
        case 'damadics'
            settings.RMSETarget = .7764409465751495;
            settings.Fit.CoefficientLower = 1e-6;
            settings.Fit.CoefficientUpper = 1e4;
            settings.Fit.GainLower = 1e-6;
            settings.Fit.GainUpper = 1e4;
            settings.Fit.StabilityMargin = 1e-10;
        case 'pt326'
            settings.ProfileInitialState = true;
            settings.MinimumTrusted = 20;
            settings.MaskedSentinel = 0;
            settings.WarmFixedStart = 1;
            settings.RMSETarget = .17460273682266203;
            settings.MAD.ResidualZeroTolerance = 1e-12;
            settings.Fit = struct('Starts', [.1927225579255227 .9580371749284126 .3205024735158558; .5 2 .5], ...
                'Lower', 1e-6, 'Upper', 1000, 'FDStep', 1e-5, 'MaxIterations', 150, ...
                'MaxEvaluations', 1800, 'FunctionTolerance', 1e-10, 'StepTolerance', 1e-9, ...
                'OptimalityTolerance', 1e-8, 'StabilityMargin', 1e-8, ...
                'JacobianConditionLimit', 1e8, 'BoundaryLogMargin', .0001);
    end
end

function fitResult = fit_fixed_initial_state(inputData, outputValues, trusted, settings, previousParameters)
    fitSettings = settings.Fit;
    fittingIndices = find(trusted);
    observed = outputValues(fittingIndices);
    parameterCount = inputData.Order + inputData.NumeratorOrder + 1;
    sampleCount = inputData.Plan.N;
    initialParameters = fitSettings.Starts;
    if ~isempty(previousParameters)
        initialParameters = [previousParameters(:).'; initialParameters(settings.WarmFixedStart,:)];
    end
    logLowerBounds = log([fitSettings.CoefficientLower * ones(inputData.Order, 1); ...
        fitSettings.GainLower * ones(inputData.NumeratorOrder + 1, 1)]);
    logUpperBounds = log([fitSettings.CoefficientUpper * ones(inputData.Order, 1); ...
        fitSettings.GainUpper * ones(inputData.NumeratorOrder + 1, 1)]);
    scale = inputData.Plan.Dt ^ (-inputData.Order);
    solverOptions = optimoptions('lsqnonlin', 'Display', 'off', ...
        'FiniteDifferenceType', 'central', 'FiniteDifferenceStepSize', fitSettings.FDStep, ...
        'MaxIterations', fitSettings.MaxIterations, 'MaxFunctionEvaluations', fitSettings.MaxEvaluations, ...
        'FunctionTolerance', fitSettings.FunctionTolerance, 'StepTolerance', fitSettings.StepTolerance, ...
        'OptimalityTolerance', fitSettings.OptimalityTolerance);
    starts = repmat(struct('Theta', [], 'ExitFlag', -99, 'Objective', Inf, 'Valid', false), ...
        size(initialParameters, 1), 1);
    for startIndex = 1:size(initialParameters, 1)
        [logParameters, ~, ~, exitFlag] = lsqnonlin(@residual, ...
            log(initialParameters(startIndex,:).'), logLowerBounds, logUpperBounds, solverOptions);
        parameters = exp(logParameters);
        [predictedOutput, valid] = evaluate(parameters);
        starts(startIndex).Theta = parameters;
        starts(startIndex).ExitFlag = exitFlag;
        starts(startIndex).Valid = valid && all(logParameters >= logLowerBounds - 1e-9) ...
            && all(logParameters <= logUpperBounds + 1e-9);
        if starts(startIndex).Valid
            starts(startIndex).Objective = sum((observed - predictedOutput(fittingIndices)) .^ 2);
        end
    end
    validStarts = find([starts.Valid] & [starts.ExitFlag] > 0);
    if isempty(validStarts)
        validStarts = find([starts.Valid]);
    end
    fitResult = struct('Theta', nan(parameterCount, 1), 'InitialState', inputData.InitialState, ...
        'Prediction', nan(sampleCount, 1), 'Targets', fittingIndices, 'RMSE', Inf, ...
        'Success', false, 'Failure', 'all_starts_failed');
    if isempty(validStarts)
        return;
    end
    [~, bestIndex] = min([starts(validStarts).Objective]);
    bestStart = validStarts(bestIndex);
    parameterVector = starts(bestStart).Theta;
    [predictedOutput, isValid] = evaluate(parameterVector);
    sensitivityMatrix = zeros(numel(fittingIndices), parameterCount);
    for parameterIndex = 1:parameterCount
        differenceStep = fitSettings.FDStep * max(1, abs(parameterVector(parameterIndex)));
        plusParameters = parameterVector;
        minusParameters = parameterVector;
        plusParameters(parameterIndex) = plusParameters(parameterIndex) + differenceStep;
        minusParameters(parameterIndex) = minusParameters(parameterIndex) - differenceStep;
        plusPrediction = evaluate(plusParameters);
        minusPrediction = evaluate(minusParameters);
        sensitivityMatrix(:, parameterIndex) = ...
            (plusPrediction(fittingIndices) - minusPrediction(fittingIndices)) / (2 * differenceStep);
    end
    singularValues = svd(sensitivityMatrix, 'econ');
    sensitivityRank = rank(sensitivityMatrix);
    sensitivityCondition = Inf;
    if sensitivityRank == parameterCount
        sensitivityCondition = singularValues(1) / singularValues(end);
    end
    model = state_model(parameterVector);
    isStable = all(real(eig(model.A)) < -fitSettings.StabilityMargin);
    isAtBoundary = any(abs(log(parameterVector) - logLowerBounds) < 1e-4 ...
        | abs(log(parameterVector) - logUpperBounds) < 1e-4);
    fitResult.Theta = parameterVector;
    fitResult.Prediction = predictedOutput;
    fitResult.RMSE = sqrt(sum((observed - predictedOutput(fittingIndices)) .^ 2) / numel(fittingIndices));
    fitResult.Failure = '';
    if starts(bestStart).ExitFlag <= 0 || ~isValid
        fitResult.Failure = 'optimizer_failed';
    elseif ~isStable
        fitResult.Failure = 'unstable';
    elseif isAtBoundary
        fitResult.Failure = 'parameter_boundary';
    elseif sensitivityRank < parameterCount || ~isfinite(sensitivityCondition) ...
            || sensitivityCondition > fitSettings.JacobianConditionLimit
        fitResult.Failure = 'identifiability_failed';
    end
    fitResult.Success = isempty(fitResult.Failure);

    function [prediction, valid] = evaluate(parameters)
        model = state_model(parameters);
        segments = input_segments(model, inputData.Plan);
        prediction = propagate_state(model, segments, inputData.InitialState);
        valid = all(isfinite(prediction)) && all(real(eig(model.A)) < -fitSettings.StabilityMargin);
    end

    function residualValues = residual(logParameters)
        if any(~isfinite(logParameters)) || any(logParameters < logLowerBounds) ...
                || any(logParameters > logUpperBounds)
            residualValues = 1e8 * (1 + norm(max(logLowerBounds - logParameters, 0) ...
                + max(logParameters - logUpperBounds, 0))) * ones(numel(fittingIndices), 1);
            return;
        end
        parameters = exp(logParameters);
        model = state_model(parameters);
        maximumPole = max(real(eig(model.A)));
        if maximumPole >= -fitSettings.StabilityMargin
            residualValues = 1e8 * (1 + maximumPole + fitSettings.StabilityMargin) ...
                * ones(numel(fittingIndices), 1);
            return;
        end
        [prediction, valid] = evaluate(parameters);
        if ~valid
            residualValues = 1e8 * ones(numel(fittingIndices), 1);
            return;
        end
        residualValues = scale * (observed - prediction(fittingIndices));
    end
end

function fitResult = fit_profiled_initial_state(inputData, outputValues, trusted, settings, previousParameters)
    fitSettings = settings.Fit;
    fittingIndices = find(trusted);
    observed = outputValues(fittingIndices);
    modelOrder = inputData.Order;
    parameterCount = modelOrder + inputData.NumeratorOrder + 1;
    initialParameters = fitSettings.Starts;
    if ~isempty(previousParameters)
        initialParameters = [previousParameters(:).'; initialParameters(1,:)];
    end
    logLowerBounds = log(fitSettings.Lower * ones(parameterCount, 1));
    logUpperBounds = log(fitSettings.Upper * ones(parameterCount, 1));
    starts = repmat(struct('Theta', [], 'ExitFlag', -99, 'Objective', Inf, 'Valid', false), ...
        size(initialParameters, 1), 1);
    solverOptions = optimoptions('lsqnonlin', 'Display', 'off', ...
        'FiniteDifferenceType', 'central', 'FiniteDifferenceStepSize', fitSettings.FDStep, ...
        'MaxIterations', fitSettings.MaxIterations, 'MaxFunctionEvaluations', fitSettings.MaxEvaluations, ...
        'FunctionTolerance', fitSettings.FunctionTolerance, 'StepTolerance', fitSettings.StepTolerance, ...
        'OptimalityTolerance', fitSettings.OptimalityTolerance);
    for startIndex = 1:size(initialParameters, 1)
        [logParameters, ~, ~, exitFlag] = lsqnonlin(@residual, ...
            log(initialParameters(startIndex,:).'), logLowerBounds, logUpperBounds, solverOptions);
        parameters = exp(logParameters);
        [prediction, ~, ~, valid] = evaluate(parameters);
        starts(startIndex).Theta = parameters;
        starts(startIndex).ExitFlag = exitFlag;
        starts(startIndex).Valid = valid && all(logParameters >= logLowerBounds - 1e-8) ...
            & all(logParameters <= logUpperBounds + 1e-8);
        if starts(startIndex).Valid
            starts(startIndex).Objective = sum((observed - prediction(fittingIndices)) .^ 2);
        end
    end
    validStarts = find([starts.Valid] & [starts.ExitFlag] > 0);
    if isempty(validStarts)
        validStarts = find([starts.Valid]);
    end
    fitResult = struct('Theta', nan(parameterCount, 1), 'InitialState', nan(modelOrder, 1), ...
        'Prediction', nan(size(outputValues)), 'Targets', fittingIndices, 'RMSE', Inf, ...
        'Success', false, 'Failure', 'all_starts_failed');
    if isempty(validStarts)
        return;
    end
    [~, bestIndex] = min([starts(validStarts).Objective]);
    bestStart = validStarts(bestIndex);
    parameterVector = starts(bestStart).Theta;
    [prediction, initialState, stateBasis, valid] = evaluate(parameterVector);
    sensitivityMatrix = zeros(numel(fittingIndices), parameterCount + modelOrder);
    for parameterIndex = 1:parameterCount
        differenceStep = fitSettings.FDStep;
        plusParameters = parameterVector;
        minusParameters = parameterVector;
        plusParameters(parameterIndex) = plusParameters(parameterIndex) * exp(differenceStep);
        minusParameters(parameterIndex) = minusParameters(parameterIndex) * exp(-differenceStep);
        [plusForced, plusBasis] = zoh_response(plusParameters, inputData);
        [minusForced, minusBasis] = zoh_response(minusParameters, inputData);
        sensitivityMatrix(:, parameterIndex) = ...
            ((plusForced(fittingIndices) + plusBasis(fittingIndices,:) * initialState) ...
            - (minusForced(fittingIndices) + minusBasis(fittingIndices,:) * initialState)) ...
            / (2 * differenceStep);
    end
    sensitivityMatrix(:, parameterCount + 1:end) = stateBasis(fittingIndices,:);
    columnNorms = sqrt(sum(sensitivityMatrix .^ 2, 1));
    normalizedSensitivity = sensitivityMatrix ./ max(columnNorms, realmin);
    singularValues = svd(normalizedSensitivity, 'econ');
    sensitivityRank = rank(normalizedSensitivity);
    sensitivityCondition = Inf;
    if sensitivityRank == parameterCount + modelOrder
        sensitivityCondition = singularValues(1) / singularValues(end);
    end
    model = state_model(parameterVector);
    isStable = all(real(eig(model.A)) < -fitSettings.StabilityMargin);
    isAtBoundary = any(abs(log(parameterVector) - logLowerBounds) < fitSettings.BoundaryLogMargin ...
        | abs(log(parameterVector) - logUpperBounds) < fitSettings.BoundaryLogMargin);
    fitResult.Theta = parameterVector;
    fitResult.InitialState = initialState;
    fitResult.Prediction = prediction;
    fitResult.RMSE = sqrt(sum((observed - prediction(fittingIndices)) .^ 2) / numel(fittingIndices));
    fitResult.Failure = '';
    if starts(bestStart).ExitFlag <= 0 || ~valid
        fitResult.Failure = 'optimizer_failed';
    elseif ~isStable
        fitResult.Failure = 'unstable';
    elseif isAtBoundary
        fitResult.Failure = 'parameter_boundary';
    elseif sensitivityRank < parameterCount + modelOrder || ~isfinite(sensitivityCondition) ...
            || sensitivityCondition > fitSettings.JacobianConditionLimit
        fitResult.Failure = 'identifiability_failed';
    end
    fitResult.Success = isempty(fitResult.Failure);

    function [prediction, initialState, stateBasis, valid] = evaluate(parameters)
        [forced, stateBasis, model] = zoh_response(parameters, inputData);
        initialState = stateBasis(fittingIndices,:) \ (observed - forced(fittingIndices));
        prediction = forced + stateBasis * initialState;
        valid = all(isfinite(prediction)) && all(isfinite(initialState)) ...
            && all(real(eig(model.A)) < -fitSettings.StabilityMargin);
    end

    function residualValues = residual(logParameters)
        if any(~isfinite(logParameters)) || any(logParameters < logLowerBounds) ...
                || any(logParameters > logUpperBounds)
            residualValues = 1e8 * ones(numel(fittingIndices), 1);
            return;
        end
        [prediction, ~, ~, valid] = evaluate(exp(logParameters));
        if valid
            residualValues = observed - prediction(fittingIndices);
        else
            residualValues = 1e8 * ones(numel(fittingIndices), 1);
        end
    end
end

function model = state_model(parameters)
    A = [0 1; -parameters(1) -parameters(2)];
    if numel(parameters) == 4
        B = [parameters(3); parameters(4) - parameters(2) * parameters(3)];
    else
        B = [0; parameters(3)];
    end
    model = struct('A', A, 'B', B, 'C', [1 0], 'Order', 2, ...
        'NumeratorOrder', numel(parameters) - 3);
end

function segments = input_segments(model, plan)
    transitionMatrix = expm(model.A * plan.Dt);
    substepWidth = plan.Dt / plan.Substeps;
    substepTransition = expm(model.A * substepWidth);
    quadratureWeights = zeros(2, plan.Substeps + 1);
    accumulatedTransition = eye(2);
    for quadratureIndex = plan.Substeps:-1:0
        quadratureWeight = substepWidth;
        if quadratureIndex == 0 || quadratureIndex == plan.Substeps
            quadratureWeight = substepWidth / 2;
        end
        quadratureWeights(:, quadratureIndex + 1) = quadratureWeight * accumulatedTransition * model.B;
        accumulatedTransition = accumulatedTransition * substepTransition;
    end
    segments = struct('P', transitionMatrix, 'Eta', quadratureWeights * plan.Values, 'N', plan.N);
end

function predictedOutput = propagate_state(model, segments, initialState)
    stateHistory = zeros(2, segments.N);
    stateHistory(:, 1) = initialState;
    for sampleIndex = 2:segments.N
        stateHistory(:, sampleIndex) = segments.P * stateHistory(:, sampleIndex - 1) ...
            + segments.Eta(:, sampleIndex - 1);
    end
    predictedOutput = (model.C * stateHistory).';
end

function [forcedOutput, stateBasis, model] = zoh_response(parameters, inputData)
    model = state_model(parameters);
    sampleCount = numel(inputData.Time);
    interval = inputData.Time(2) - inputData.Time(1);
    augmentedTransition = expm([model.A, model.B; zeros(1, 3)] * interval);
    stateTransition = augmentedTransition(1:2, 1:2);
    inputTransition = augmentedTransition(1:2, end);
    delay = inputData.DelaySamples;
    delayedInput = [inputData.InputPrehistory(end - delay + 1:end); inputData.U(1:end - delay)];
    forcedOutput = zeros(sampleCount, 1);
    stateBasis = zeros(sampleCount, 2);
    forcedState = zeros(2, 1);
    initialStateTransition = eye(2);
    for sampleIndex = 1:sampleCount
        forcedOutput(sampleIndex) = model.C * forcedState;
        stateBasis(sampleIndex,:) = model.C * initialStateTransition;
        forcedState = stateTransition * forcedState + inputTransition * delayedInput(sampleIndex);
        initialStateTransition = stateTransition * initialStateTransition;
    end
end

function candidateIndices = terminal_review(parameterVector, time, outputValues, basePrediction, candidateIndices, ...
    modelOrder, settings)
    model = state_model(parameterVector);
    sampleCount = numel(outputValues);
    outputTransitionRows = zeros(sampleCount, modelOrder);
    for lag = 0:sampleCount - 1
        outputTransitionRows(lag + 1,:) = model.C * expm(model.A * (time(lag + 1) - time(1)));
    end
    trusted = true(sampleCount, 1);
    trusted(candidateIndices) = false;
    remaining = ~trusted;
    referenceResiduals = nan(sampleCount, 1);
    selectedIndices = find(trusted);
    for referencePosition = 1:numel(selectedIndices)
        targetIndex = selectedIndices(referencePosition);
        historyIndices = selectedIndices(max(1, referencePosition - modelOrder):referencePosition - 1);
        [predictedOutput, isValid] = history_prediction(targetIndex, historyIndices, outputValues, basePrediction, ...
            outputTransitionRows, modelOrder, settings);
        if isValid
            referenceResiduals(targetIndex) = outputValues(targetIndex) - predictedOutput;
        end
    end
    for targetIndex = sort(candidateIndices(:)).'
        referenceValues = referenceResiduals(isfinite(referenceResiduals));
        historyIndices = find(trusted & (1:sampleCount).' < targetIndex, modelOrder, 'last');
        [predictedOutput, isValid] = history_prediction(targetIndex, historyIndices, outputValues, basePrediction, ...
            outputTransitionRows, modelOrder, settings);
        if ~isValid || numel(referenceValues) < settings.MinimumReferenceCount
            continue;
        end
        referenceMean = mean(referenceValues);
        referenceStd = std(referenceValues, 0);
        if ~isfinite(referenceMean) || ~isfinite(referenceStd)
            continue;
        end
        if abs(outputValues(targetIndex) - predictedOutput - referenceMean) <= settings.Multiplier * referenceStd
            trusted(targetIndex) = true;
            remaining(targetIndex) = false;
            updatedIndices = [targetIndex; find(trusted & (1:sampleCount).' > targetIndex, modelOrder, ...
                'first')];
            for targetIndex = updatedIndices(:).'
                historyIndices = find(trusted & (1:sampleCount).' < targetIndex, modelOrder, 'last');
                [predictedOutput, isValid] = history_prediction(targetIndex, historyIndices, outputValues, ...
                    basePrediction, outputTransitionRows, modelOrder, settings);
                referenceResiduals(targetIndex) = NaN;
                if isValid
                    referenceResiduals(targetIndex) = outputValues(targetIndex) - predictedOutput;
                end
            end
        end
    end
    candidateIndices = find(remaining);
end

function [predictedOutput, isValid] = history_prediction(targetIndex, historyIndices, outputValues, basePrediction, ...
    outputTransitionRows, modelOrder, settings)
    predictedOutput = NaN;
    isValid = false;
    if numel(historyIndices) < modelOrder
        return;
    end
    anchorIndex = historyIndices(1);
    observabilityMatrix = outputTransitionRows(historyIndices - anchorIndex + 1,:);
    outputRow = outputTransitionRows(targetIndex - anchorIndex + 1,:);
    if rank(observabilityMatrix) < modelOrder || ~isfinite(cond(observabilityMatrix)) || cond(observabilityMatrix) > settings.ConditionLimit || ...
        rcond(observabilityMatrix) < settings.RcondMinimum
        return;
    end
    stateCorrection = observabilityMatrix \ (outputValues(historyIndices) - basePrediction(historyIndices));
    predictedOutput = basePrediction(targetIndex) + outputRow * stateCorrection;
    isValid = isfinite(predictedOutput);
end
