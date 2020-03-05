package ocrai;

import java.util.Random;

public class NetworkOperations {

    static NeuralNetworkDataSet prepareDataForNeuralNetwork(double[][] dataSet, String dataSetLabel) {
        NeuralNetworkDataSet preppedDataSet = new NeuralNetworkDataSet(64, 10, dataSetLabel);

        for (double[] trainDatum : dataSet) {
            double[] input = new double[64];
            // 10 outputs as 10 labels. Each output refers to a specific label
            double[] output = new double[10];
            double label = trainDatum[64];
            // Marking the output label as 1.0 in the output
            output[(int) label] = 1d;
            // Dividing by 16 to get them to be numbers between 0-1
            for (int i = 0; i < 64; i++) {
                input[i] = trainDatum[i] / (double) 16;
            }
            preppedDataSet.loadData(input, output);
        }
        return preppedDataSet;
    }

    public static double[][] assignRandomWeights
            (int currentLayerNeuronCount, int previousLayerNeuronCount, double minimum, double maximum) {
        if (currentLayerNeuronCount < 1 || previousLayerNeuronCount < 1) return null;
        double[][] weights = new double[currentLayerNeuronCount][previousLayerNeuronCount];
        for (int i = 0; i < currentLayerNeuronCount; i++) {
            // Every Current Neuron in the layer has a weight assigned to it.
            weights[i] = new Random().doubles(previousLayerNeuronCount, minimum, maximum).toArray();
        }
        return weights;
    }

    // Gets the index of the highest value
    public static int indexOfMax(double[] a) {
        int maxAt = 0;
        for (int i = 0; i < a.length; i++) {
            maxAt = a[i] > a[maxAt] ? i : maxAt;
        }
        return maxAt;
    }
}
