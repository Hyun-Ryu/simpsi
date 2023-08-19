root = '/';

%% Parameters
len_patch = 32;         % Length of patch
sampsPerSym = 4;        % Upsampling factor

modulationTypesStr = '32FSK  ';
modulationTypes = 32;
numModulationTypes = length(modulationTypes);

noiseSnrList = 10:2:28;
lenNoiseSnrList = length(noiseSnrList);

%% Data Generation

for M = modulationTypes
    numPatch = 288;
    numInstance = numPatch;

    for noiseSnr = noiseSnrList
        xt = zeros(numInstance, len_patch);
        yt = zeros(numInstance, len_patch*sampsPerSym);
        cnt = 0;

        for patch = 1:numPatch
            cnt = cnt + 1;

            % Generate random data
            x = randi([0 M-1],len_patch,1);

            % Modulate
            freqsep = 4; fs = 128;
            y = fskmod(x, M, freqsep, sampsPerSym, fs);

            % AWGN channel
            yc = awgn(y, noiseSnr);

            % Normalize average power to 1
            avg_pow = sum(abs(yc).^2)/length(yc);
            yc = yc / sqrt(avg_pow);

            % Store in array
            xt(cnt,:) = x;
            yt(cnt,:) = yc;
        end

        % Save data
        save(sprintf('%s%sdB.mat',root,int2str(noiseSnr)), 'yt', 'xt')

    end
end