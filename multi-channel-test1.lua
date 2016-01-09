torch = require 'torch'
nn = require 'nn'
local hdf5 = require 'hdf5'
local totem = require 'totem'
----tester = require totem.--tester()


filepath = '/Users/aleksandrspiridonov/Box Sync/code/ds_fin1/outdir/TorchData.h5'
print(filepath)
-- some params for conv and sampling layers

kWC1 = 2
nCO1 = 16
kWC2 = 5
nCO2 = 128
kWM = 2
dWM = 2



-- open h5 file and load data

dataFile = hdf5.open(filepath,'r')
--tester:assertne(dataFile, nil, "hdf5.open returned nil")


params = dataFile:read('scriptParams'):all()
nTrain = params[1]
--tester:assertne(nTrain, nil, "nTrain is nil")
nTest = params[2]
--tester:assertne(nTest, nil, "nTest is nil")
nChannelSamples = params[3]
--tester:assertne(nChannelSamples, nil, "nChannelSamples is nil")
nInputChannels = params[4]
--tester:assertne(nInputChannels, nil, "nInputChannels is nil")
nOutputChannels = params[5]
--tester:assertne(nOutputChannels, nil, "nOutputChannels is nil")

dataFile:close()

-- if nChannelSamples = 25
nSCO1 = (nChannelSamples - kWC1) / 1 + 1		-- (25-2)/1 + 1 = 24
nSM1 = (nSCO1 - kWM) / dWM + 1	-- (24-2)/2 + 1 = 12
nSCO2 = (nSM1 - kWC2) / 1 + 1	-- (12-5)/1 + 1 = 8
nSM2 = (nSCO2 - kWM) / dWM + 1	-- (8-2)/2 + 1  = 4




-- assemble the nnet

-- assemble the parallel convolution channels (pcc)
plt = nn.ParallelTable()
pcc = {}
for i = 1, nInputChannels do
	pcc[i] = nn.Sequential()
	pcc[i]:add(nn.Reshape(nChannelSamples,1))
	pcc[i]:add(nn.TemporalConvolution(1,nCO1,kWC1,1))
	pcc[i]:add(nn.TemporalMaxPooling(kWM,dWM))
	pcc[i]:add(nn.TemporalConvolution(nCO1,nCO2,kWC2,1))
	pcc[i]:add(nn.TemporalMaxPooling(kWM,dWM))
	pcc[i]:add(nn.View(nCO2*nSM2))
	--pcc[i]:add(nn.Dropout())
	pcc[i]:add(nn.Linear(nCO2*nSM2,nCO2))
	pcc[i]:add(nn.Tanh())
	--pcc[i]:add(nn.Linear(nChannelSamples,1))
	plt:add(pcc[i])
end

mlp = nn.Sequential()
mlp:add(nn.SplitTable(1))
mlp:add(plt) -- the parallel portion
mlp:add(nn.JoinTable(1))
mlp:add(nn.Dropout())
mlp:add(nn.Linear(nInputChannels*nCO2, nOutputChannels))


--criterion = nn.MSECriterion()
criterion = nn.AbsCriterion()
--criterion.sizeAverage = false 

--generate dataset from h5
dataFile = hdf5.open(filepath,'r')


dataset={};
function dataset:size() return nTrain end -- 100 examples
for i=1,dataset:size() do 
  local input = dataFile:read('trainSet/in'..tostring(i-1)):all();
  local output = dataFile:read('trainSet/out'..tostring(i-1)):all();
  --print(input)
  dataset[i] = {input, output}
end

dataFile:close()
-- train net
 
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)




dataFile = hdf5.open(filepath,'r')


--dataset={};
--function dataset:size() return nTrain end -- 100 examples
for i=1,nTest do 
  local input = dataFile:read('testSet/in'..tostring(i-1)):all();
  local output = dataFile:read('testSet/out'..tostring(i-1)):all();
  print(torch.cat(output,mlp:forward(input),2))
  --dataset[i] = {input, output}
end

dataFile:close()

