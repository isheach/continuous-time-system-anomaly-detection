function [simulationData, modelInputs] = generate_simulation_data
    design = struct('sample_interval', .01, 'end_time', 10, ...
        'denominator_descending', [1; 1.8; .72], 'numerator', [.6; 1.2], ...
        'outlier_counts', [30; 50; 100], 'amplitude_multipliers', [.25; .5; 1; 2], ...
        'position_seed', 2026101801, 'amplitude_seed', 2026101802, 'sign_seed', 2026101803);
    design.input_cases = {struct('breaks', [0 1 2.2 3.5 5 6.2 7.5 9], ...
        'levels', [.4 1 -.6 .8 -.3 1.2 .1 -.8]), ...
        struct('frequencies_rad_s', [.9 2.2], 'amplitudes', [.8 .5], 'phases_rad', [0 0], 'offset', 0)};
    sampleTimes = (0:design.sample_interval:design.end_time).';
    trueParameters = [flipud(design.denominator_descending(2:end)); design.numerator(:)];
    inputCases = cell(2, 1);
    modelInputs = cell(2, 1);
    for inputIndex = 1:2
        spec = design.input_cases{inputIndex};
        if inputIndex == 1
            inputDefinition = struct('Kind', 'piecewise', 'Breaks', spec.breaks(:).', 'Levels', spec.levels(:).', ...
                'Frequencies', [], 'Amplitudes', [], 'Phases', [], 'Offset', 0);
        else
            inputDefinition = struct('Kind', 'sine', 'Breaks', [], 'Levels', [], 'Frequencies', spec.frequencies_rad_s(:).', ...
                'Amplitudes', spec.amplitudes(:).', 'Phases', spec.phases_rad(:).', 'Offset', spec.offset);
        end
        [cleanOutput, referenceStates, backend] = clean_response(trueParameters, 2, inputDefinition, sampleTimes);
        integrationPlan = integration_plan(inputDefinition, sampleTimes, 64);
        modelInputs{inputIndex} = struct('Order', 2, 'NumeratorOrder', 1, 'Time', sampleTimes, 'InitialState', ...
            zeros(2, 1), 'Plan', integrationPlan);
        sampled = sample_input(inputDefinition, sampleTimes);
        if inputIndex == 1
            sampled = [integrationPlan.Values(1,:).'; inputDefinition.Levels(end)];
        end
        inputCases{inputIndex} = struct('ID', inputIndex, 'Theta', trueParameters, 'Time', sampleTimes, ...
            'Input', inputDefinition, 'SampledU', sampled, 'CleanY', cleanOutput, 'ReferenceState', referenceStates, ...
            'Generator', backend);
    end
    records = struct('ID', {}, 'CaseID', {}, 'Trial', {}, 'Count', {}, 'Multiplier', {}, 'Y', {}, 'Indices', ...
        {}, 'Added', {});
    baseMagnitudes = (100:5:380).' / 1000;
    for inputIndex = 1:2
        for trialIndex = 1:100
            positionStream = RandStream('mt19937ar', 'Seed', design.position_seed + 10000 * inputIndex + trialIndex);
            amplitudeStream = RandStream('mt19937ar', 'Seed', design.amplitude_seed + 10000 * inputIndex + trialIndex);
            signStream = RandStream('mt19937ar', 'Seed', design.sign_seed + 10000 * inputIndex + trialIndex);
            positionOrder = randperm(positionStream, numel(sampleTimes), 100).';
            signedMagnitudes = baseMagnitudes(randi(amplitudeStream, numel(baseMagnitudes), 100, 1)) .* (2 * (rand(signStream, ...
                100, 1) >= .5) - 1);
            for anomalyCount = design.outlier_counts(:).'
                for amplitudeMultiplier = design.amplitude_multipliers(:).'
                    indices = positionOrder(1:anomalyCount);
                    added = signedMagnitudes(1:anomalyCount) * amplitudeMultiplier;
                    cleanOutput = inputCases{inputIndex}.CleanY;
                    cleanOutput(indices) = cleanOutput(indices) + added;
                    records(end + 1) = struct('ID', numel(records) + 1, 'CaseID', inputIndex, 'Trial', ...
                        trialIndex, 'Count', anomalyCount, 'Multiplier', amplitudeMultiplier, 'Y', cleanOutput, ...
                        'Indices', indices, 'Added', added);
                end
            end
        end
    end
    simulationData = struct('Cases', {inputCases}, 'Records', records);
end

function [cleanOutput, referenceStates, backend] = clean_response(trueParameters, modelOrder, inputData, ...
    sampleTimes)
    trueParameters = trueParameters(:);
    numeratorOrder = numel(trueParameters) - modelOrder - 1;
    dt = sampleTimes(2) - sampleTimes(1);
    sampleCount = numel(sampleTimes);
    generatorA = zeros(modelOrder);
    generatorA(1:modelOrder - 1, 2:modelOrder) = eye(modelOrder - 1);
    generatorA(modelOrder,:) = - trueParameters(1:modelOrder).';
    generatorB = zeros(modelOrder, 1);
    generatorB(end) = 1;
    generatorC = zeros(1, modelOrder);
    generatorC(1:numeratorOrder + 1) = flipud(trueParameters(modelOrder + 1:end)).';
    stateHistory = zeros(modelOrder, sampleCount);
    if strcmp(inputData.Kind, 'sine')
        frequencyCount = numel(inputData.Frequencies);
        augmentedMatrix = zeros(modelOrder + 2 * frequencyCount + 1);
        augmentedMatrix(1:modelOrder, 1:modelOrder) = generatorA;
        augmentedState = zeros(modelOrder + 2 * frequencyCount + 1, 1);
        augmentedState(end) = 1;
        augmentedMatrix(1:modelOrder, end) = generatorB * inputData.Offset;
        for frequencyIndex = 1:frequencyCount
            oscillatorIndex = modelOrder + 2 * frequencyIndex - 1;
            angularFrequency = inputData.Frequencies(frequencyIndex);
            augmentedMatrix(oscillatorIndex:oscillatorIndex + 1, oscillatorIndex:oscillatorIndex + 1) = [0, angularFrequency; - angularFrequency, 0];
            augmentedMatrix(1:modelOrder, oscillatorIndex) = generatorB * inputData.Amplitudes(frequencyIndex);
            augmentedState(oscillatorIndex:oscillatorIndex + 1) = [sin(inputData.Phases(frequencyIndex)); cos(inputData.Phases(frequencyIndex))];
        end
        transitionMatrix = expm(augmentedMatrix * dt);
        for sampleIndex = 2:sampleCount
            augmentedState = transitionMatrix * augmentedState;
            stateHistory(:, sampleIndex) = augmentedState(1:modelOrder);
        end
    else
        transitionMatrix = expm([generatorA, generatorB; zeros(1, modelOrder + 1)] * dt);
        inputValues = sample_input(inputData, (sampleTimes(1:end - 1) + sampleTimes(2:end)) / 2);
        for sampleIndex = 2:sampleCount
            stateHistory(:, sampleIndex) = transitionMatrix(1:modelOrder, 1:modelOrder) * stateHistory(:, ...
                sampleIndex - 1) + transitionMatrix(1:modelOrder, end) * inputValues(sampleIndex - 1);
        end
    end
    cleanOutput = (generatorC * stateHistory).';
    observabilityMatrix = zeros(modelOrder);
    for frequencyIndex = 1:modelOrder
        observabilityMatrix(frequencyIndex,:) = generatorC * generatorA ^ (frequencyIndex - 1);
    end
    referenceStates = observabilityMatrix * stateHistory;
    backend = 'independent controllable realization and exact augmented exponential';
end

function integrationPlan = integration_plan(inputData, sampleTimes, substeps)
    sampleInterval = sampleTimes(2) - sampleTimes(1);
    substepFractions = (0:substeps).' / substeps;
    integrationTimes = sampleTimes(1:end - 1).' + sampleInterval * substepFractions;
    inputSamples = sample_input(inputData, integrationTimes);
    if strcmp(inputData.Kind, 'piecewise')
        interiorValues = sample_input(inputData, (sampleTimes(1:end - 1) + sampleTimes(2:end)) / 2);
        inputSamples = repmat(interiorValues.', substeps + 1, 1);
    end
    integrationPlan = struct('Dt', sampleInterval, 'Substeps', substeps, 'Values', inputSamples, 'Time', ...
        sampleTimes, 'Input', inputData, 'N', numel(sampleTimes));
end

function inputValues = sample_input(inputData, queryTimes)
    if strcmp(inputData.Kind, 'sine')
        inputValues = inputData.Offset + zeros(size(queryTimes));
        for componentIndex = 1:numel(inputData.Frequencies)
            inputValues = inputValues + inputData.Amplitudes(componentIndex) * sin(inputData.Frequencies(componentIndex) * queryTimes + inputData.Phases(componentIndex));
        end
    else
        inputValues = zeros(size(queryTimes));
        for componentIndex = 1:numel(inputData.Breaks)
            inputValues(queryTimes >= inputData.Breaks(componentIndex)) = inputData.Levels(componentIndex);
        end
    end
end
