package ocrai;

import java.util.ArrayList;

public class NeuralNetworkDataSet {
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final String LABEL;
    private ArrayList<double[][]> data = new ArrayList<>();

    public NeuralNetworkDataSet(int INPUT_SIZE, int OUTPUT_SIZE, String LABEL) {
        this.INPUT_SIZE = INPUT_SIZE;
        this.OUTPUT_SIZE = OUTPUT_SIZE;
        this.LABEL = LABEL;
    }

    public int getSize() {
        return data.size();
    }

    public double[] getInput(int index) {
        return data.get(index)[0];
    }

    public double[] getOutput(int index) {
        return data.get(index)[1];
    }

    public void loadData(double[] input, double[] output) {
        data.add(new double[][]
                {input, output});
    }
}
