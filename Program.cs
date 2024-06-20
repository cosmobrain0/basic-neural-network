Random generator = new Random((int)(DateTime.Now.Ticks << 32 >> 32));
Random[] generators = new Random[256];
for (int i=0; i<generators.Length; i++) generators[i] = new Random((int)(generator.NextInt64() << 32 >> 32));
NeuralNetwork[] networks = new NeuralNetwork[10000];

for (int i=0; i<networks.Length; i++)
	networks[i] = new NeuralNetwork(2, new int[] { 2, 2 }, 2, generator, true);

// trying to learn to differentiate between stuff below "y = x^2" and stuff above it
Func<float[], float[]> idealOutputs = inputs => inputs[1] > inputs[0] ? new float[] { 1, 0 } : new float[] { 0, 1 };

Func<float[]> randomDataPoint = () => new float[] { generator.NextSingle(), generator.NextSingle() };

Func<float[], int> highestIndex = values =>
{
	int result = 0;
	for (int i=1; i<values.Length; i++)
		if (values[i] > values[result]) result = i;
	return result;
};

(float[], float[])[] testCases = new (float[], float[])[400];
for (int j=0; j<testCases.Length; j++)
{
	float[] inputs = randomDataPoint();
	float[] outputs = idealOutputs(inputs);
	testCases[j] = (inputs, outputs);
}

// (NeuralNetwork network, float cost)[] result = networks.Select(x => (x, 0.0)).ToArray();
float[] costs = new float[networks.Length];
const int learningIterations = 6000;
for (int i=0; i<learningIterations; i++)
{
	if (i%10 == 0) Console.Write($"iteration {i+1} / {learningIterations}: ");
	// learning
	// TODO: possible optimisation - manual insertion sort here
	Parallel.For(0, networks.Length, j => {
		costs[j] = networks[j].AverageCost(testCases);
	});
	Array.Sort(costs, networks);
	if (i%10 == 0) Console.WriteLine($"Minimum cost - {costs[0]} | Average cost - {costs.Sum() / costs.Length} | Maximum cost - {costs[costs.Length-1]}");

	const float learnRate = 0.45f;
	Parallel.For(networks.Length/6, networks.Length/2 + networks.Length/6, j => {
		Random generator = generators[j%generators.Length];
		if (j < networks.Length/2)
			networks[j + networks.Length/2].Copy(networks[j]);
		for (int k=0; k<2; k++)
		{
			// pick a random node
			// then pick a random weight or bias
			// then change it a little

			int nodeIndex = (int)(generator.NextInt64() & 1);
			int weightIndex = (int)(generator.NextInt64()%3);

			// we can use it as a direct index as there are no hidden layers
			Node node = nodeIndex < 2 ? networks[j].hidden[0][nodeIndex] : nodeIndex < 4 ? networks[j].hidden[1][nodeIndex-2] : networks[j].outputs[nodeIndex-4];
			float change = (generator.NextSingle()-0.5f) * learnRate; 
			if (weightIndex == 2) node.bias += change;
			else node.incomingWeights[weightIndex] += change;
		}
		if (j < networks.Length/2)
			for (int k=0; k<2; k++)
			{
				// pick a random node
				// then pick a random weight or bias
				// then change it a little
	
				int nodeIndex = (int)(generator.NextInt64() & 1);
				int weightIndex = (int)(generator.NextInt64()%3);
	
				// we can use it as a direct index as there are no hidden layers
				Node node = networks[j + networks.Length/2].outputs[nodeIndex];
				float change = (generator.NextSingle()-0.5f) * learnRate; 
				if (weightIndex == 2) node.bias += change;
				else node.incomingWeights[weightIndex] += change;
			}
	});
}

// debug final info
Console.WriteLine("DEBUG ===");
Console.WriteLine($"Output 0 : {networks[0].outputs[0].incomingWeights[0]} | {networks[0].outputs[0].incomingWeights[1]} | bias: {networks[0].outputs[0].bias}");
Console.WriteLine($"Output 1 : {networks[0].outputs[1].incomingWeights[0]} | {networks[0].outputs[1].incomingWeights[1]} | bias: {networks[0].outputs[1].bias}");
Console.WriteLine("=========");

float minCost = float.PositiveInfinity;
(float[], float[])[] finalTestCases = new (float[], float[])[10000];
for (int j=0; j<finalTestCases.Length; j++)
{
	float[] inputs = randomDataPoint();
	float[] outputs = idealOutputs(inputs);
	finalTestCases[j] = (inputs, outputs);
}
for (int j=0; j<networks.Length; j++)
{
	minCost = Math.Min(minCost, networks[j].AverageCost(finalTestCases));
}
Console.WriteLine("min cost - " + minCost);
