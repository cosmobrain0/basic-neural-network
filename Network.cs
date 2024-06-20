public class NeuralNetwork
{
	public Node[] inputs;
	public Node[][] hidden;
	public Node[] outputs;

	public void Copy(NeuralNetwork other)
	{
		for (int x=0; x<hidden.Length; x++)
		{
			for (int y=0; y<hidden[x].Length; y++)
				hidden[x][y].Copy(other.hidden[x][y]);
		}
		for (int i=0; i<outputs.Length; i++) outputs[i].Copy(other.outputs[i]);
	}

	public NeuralNetwork(int inputSize, int[] hiddenLayerSizes, int outputSize, Random generator = null, bool random = false)
	{
		Func<float, float> activation = x => 1 / (1 + (float)Math.Exp(-x));
		inputs = new Node[inputSize].Select(_ => new Node(0, 0, activation)).ToArray();
		hidden = new Node[hiddenLayerSizes.Length][];
		for (int i=0; i<hiddenLayerSizes.Length; i++)
			hidden[i] = new Node[hiddenLayerSizes[i]]
				.Select(_ => new Node(i == 0 ? inputSize : hiddenLayerSizes[i-1], 0, activation))
				.ToArray();
		outputs = new Node[outputSize]
			.Select(_ => new Node(hiddenLayerSizes.Length > 0 ? hiddenLayerSizes[hiddenLayerSizes.Length-1] : inputSize, 0, activation))
			.ToArray();

		if (random)
		{
			for (int i=0; i<hidden.Length; i++)
			{
				for (int j=0; j<hidden[i].Length; j++)
				{
					for (int k=0; k<hidden[i][j].incomingWeights.Length; k++)
					{
						hidden[i][j].incomingWeights[k] = (generator.NextSingle()-0.5f) * 2f;
					}
					hidden[i][j].bias = (generator.NextSingle()-0.5f) * 2f;
				}
			}
			for (int i=0; i<outputs.Length; i++)
			{
				for (int j=0; j<outputs[i].incomingWeights.Length; j++)
				{
					outputs[i].incomingWeights[j] = (generator.NextSingle()-0.5f) * 2f;
				}
				outputs[i].bias = (generator.NextSingle()-0.5f) * 2f;
			}
		}
	}

	public float[] Outputs(float[] inputs)
	{
		float[] result = inputs;
		for (int i=0; i<hidden.Length; i++)
		{
			float[] newResult = new float[hidden[i].Length];
			for (int j=0; j<newResult.Length; j++)
			{
				newResult[j] = hidden[i][j].Output(result);
			}
			result = newResult;
		}
		float[] finalResult = new float[outputs.Length];
		for (int j=0; j<finalResult.Length; j++)
		{
			finalResult[j] = outputs[j].Output(result);
		}
		return finalResult;
	}

	public float AverageCost((float[] inputs, float[] idealOutputs)[] testCases)
	{
		float[] costs = new float[testCases.Length];
		Parallel.For(0, testCases.Length, i => {
			float[] dataPoint = testCases[i].inputs;
			float[] ideal = testCases[i].idealOutputs;
			costs[i] = Cost(dataPoint, ideal);
		});
		return costs.Sum() / testCases.Length;
	}

	public void Learn(Random generator, float[] inputs, float[] expectedOutputs, float learnRate)
	{
		float originalCost = Cost(inputs, expectedOutputs);
		float[][][] weightGradients = new float[hidden.Length+1][][]; // for the weights of one node

		Node[][] nodes = hidden.Concat(new Node[][] { outputs }).ToArray();
		
		for (int layerCount=0; layerCount < nodes.Length; layerCount++)
		{
			Node[] layer = nodes[layerCount];
			weightGradients[layerCount] = new float[layer.Length][];
			for (int nodeCount = 0; nodeCount < layer.Length; nodeCount++)
			{
				Node node = layer[nodeCount];
				weightGradients[layerCount][nodeCount] = new float[node.incomingWeights.Length];
				for (int i=0; i<node.incomingWeights.Length; i++)
				{
					float change = (generator.NextSingle()-0.5f) * learnRate * 2;
					node.incomingWeights[i] += change;
					float newCost = Cost(inputs, expectedOutputs);
					weightGradients[layerCount][nodeCount][i] = (newCost-originalCost)/change;
					node.incomingWeights[i] -= change;
				}
			}
		}
		float[][] biasGradients = new float[nodes.Length][];
		for (int i=0; i<nodes.Length; i++)
		{
			biasGradients[i] = new float[nodes[i].Length];
			for (int j=0; j<nodes[i].Length; j++)
			{
					float change = (generator.NextSingle()-0.5f) * learnRate * 2;
					nodes[i][j].bias += change;
					float newCost = Cost(inputs, expectedOutputs);
					biasGradients[i][j] = (newCost-originalCost)/change;
					nodes[i][j].bias -= change;
			}
		}

		for (int layerCount=0; layerCount < nodes.Length; layerCount++)
		{
			Node[] layer = nodes[layerCount];
			for (int nodeCount = 0; nodeCount < layer.Length; nodeCount++)
			{
				Node node = layer[nodeCount];
				for (int i=0; i<node.incomingWeights.Length; i++)
				{
					node.incomingWeights[i] -= weightGradients[layerCount][nodeCount][i] * learnRate;
				}
			}
		}

		for (int i=0; i<nodes.Length; i++)
		{
			for (int j=0; j<nodes[i].Length; j++)
			{
					nodes[i][j].bias -= biasGradients[i][j] * learnRate;
			}
		}
		
	}

	public float Cost(float[] inputs, float[] idealOutputs)
	{
		float[] result = Outputs(inputs);
		float sum = 0f;
		for (int i=0; i<result.Length; i++) sum += Math.Abs(idealOutputs[i]-result[i]);
		sum /= idealOutputs.Length;
		return sum;
		// return idealOutputs.Select((x, i) => Math.Abs(x-result[i])).Sum() / idealOutputs.Length;
	}
}

// TODO: some encapsulation pa-ha-leez
public class Node
{
	public float bias;
	public float[] incomingWeights;
	private Func<float, float> activation;

	public void Copy(Node other)
	{
		bias = other.bias;
		for (int i=0; i<incomingWeights.Length; i++) incomingWeights[i] = other.incomingWeights[i];
		activation = other.activation;
	}

	public bool SetWeight(int i, float weight)
	{
		if (i < 0 || i >= incomingWeights.Length) return false;
		incomingWeights[i] = weight;
		return false;
	}

	public Node(int inputs, float bias, Func<float, float> activation)
	{
		incomingWeights = new float[inputs];
		bias = 0;
		this.activation = activation;
	}

	/// <summary>
	/// NOTE: `inputs` better have the right length, or else
	/// </summary>
	public float Output(float[] inputs)
	{
		float result = 0;
		for (int i=0; i<incomingWeights.Length; i++)
		{
			result += incomingWeights[i] * inputs[i];
		}
		return activation(result / inputs.Length + bias);
	}
}

struct Connection
{
	Node start;
	Node end;
	float weight;
}
