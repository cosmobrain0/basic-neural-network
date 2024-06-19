Random generator = new Random((int)(DateTime.Now.Ticks << 32 >> 32));
Random[] generators = new Random[128];
for (int i=0; i<generators.Length; i++) generators[i] = new Random((int)(generator.NextInt64() << 32 >> 32));
NeuralNetwork[] networks = new NeuralNetwork[10000];

for (int i=0; i<networks.Length; i++)
	networks[i] = new NeuralNetwork(2, new int[] {}, 2, generator, true);

// trying to learn to differentiate between stuff below "y = 2x - 1"
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
const int learningIterations = 1000;
for (int i=0; i<learningIterations; i++)
{
	if (i%10 == 0) Console.Write($"iteration {i+1} / {learningIterations}: ");
	// learning
	// TODO: possible optimisation - manual insertion sort here
	Parallel.For(0, networks.Length, j => {
		costs[j] = networks[j].AverageCost(testCases);
	});
	// for (int j=0; j<networks.Length; j++)
	// {
	// 	float cost = networks[j].AverageCost(testCases);
	// 	costs[j] = cost;
	// 	// int k;
	// 	// for (k=j-1; k>=0; k--)
	// 	// {
	// 	// 	if (costs[k] > cost)
	// 	// 	{
	// 	// 		networks[k+1] = networks[k];
	// 	// 		costs[k+1] = costs[k];
	// 	// 	}
	// 	// 	else break;
	// 	// }
	// 	// costs[k+1] >= cost
	// 	// costs[k+1] = cost;
	// 	// networks[k+1] = network;
	// }
	Array.Sort(costs, networks);
	if (i%10 == 0) Console.WriteLine($"Minimum cost - {costs[0]}");

	const float learnRate = 0.23f;
	Parallel.For(networks.Length/3, networks.Length, j => {
		Random generator = generators[j%generators.Length];
		for (int k=0; k<4; k++)
		{
			// pick a random node
			// then pick a random weight or bias
			// then change it a little

			int nodeIndex = (int)(generator.NextInt64() & 1);
			int weightIndex = (int)(generator.NextInt64()%3);
			// int nodeIndex = 1;
			// int weightIndex = 1;

			// we can use it as a direct index as there are no hidden layers
			Node node = networks[j].outputs[nodeIndex];
			float change = (generator.NextSingle()-0.5f) * learnRate; 
			// float change = 1f * learnRate;
			if (weightIndex == 2) node.bias += change;
			else node.incomingWeights[weightIndex] += change;
		}
	});
	// for (int j=networks.Length/10; j<networks.Length; j++)
	// {
	// 	const float learnRate = 0.5f;
	// 	// pick a random node
	// 	// then pick a random weight or bias
	// 	// then change it a little
	// 	int nodeIndex = (int)(generator.NextInt64() & 1);
	// 	int weightIndex = (int)(generator.NextInt64()%3);
	// 	// we can use it as a direct index as there are no hidden layers
	// 	Node node = networks[j].outputs[nodeIndex];
	// 	float change = (generator.NextSingle()-0.5f) * 2f * learnRate; 
	// 	if (weightIndex == 2) node.bias += change;
	// 	else node.incomingWeights[weightIndex] += change;
	// }
}

// debug final info
float minCost = float.PositiveInfinity;
(float[], float[])[] finalTestCases = new (float[], float[])[100];
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
