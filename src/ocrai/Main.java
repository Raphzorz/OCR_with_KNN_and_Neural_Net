package ocrai;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.text.DecimalFormat;

public class Main {

    private final static String DATA_SET_1_URL = "Data/cw2Dataset1.csv";
    private final static String DATA_SET_2_URL = "Data/cw2Dataset2.csv";

    public final static DecimalFormat decimalFormat = new DecimalFormat("##.000");
    // The first K to start with
    private static final int initialK = 1;
    // Change this to determine how many "K's" will be tested
    private static final int numberOfK = 7;
    // <!- Set to true to run Known Nearest Neighbor -!>
    private static final boolean  RUN_KNN = true;
    // <!- Set to true to run Neural Network -!>
    private static final boolean RUN_NN = true;
    public static void main(String[] args) {

        // [] refers to the ROW in the excel sheet while the second [] refers to the column in the excel sheet
        double[] accuracy = new double[numberOfK];
        double res;
        double[][] dataSet1 = ReadFile.readFile(DATA_SET_1_URL, RUN_NN);
        double[][] dataSet2 = ReadFile.readFile(DATA_SET_2_URL, RUN_NN);
        int k = initialK;

        String dataSet1Name = null;
        try {
            dataSet1Name = Paths.get(new URI(DATA_SET_1_URL).getPath()).getFileName().toString();
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        String dataSet2Name = null;
        try {
            dataSet2Name = Paths.get(new URI(DATA_SET_2_URL).getPath()).getFileName().toString();
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }

        if (RUN_KNN) {
            System.out.println("\n<<<<<KNN>>>>>");
            for (int i = 0; i < numberOfK; i++) {
                res = KNN.getDistance(dataSet1,dataSet2,k);
                accuracy[i] = res;
                System.out.println("\nTesting on " + dataSet2Name + ": " +
                        decimalFormat.format(res) + "%" + " Accuracy using K = " + k);
                k += 2;
            }
            System.out.println("\n========== Start of second part of two-fold test ==========");
            k = initialK;
            for (int i = 0; i < numberOfK; i++) {
                res = KNN.getDistance(dataSet2,dataSet1,k);
                accuracy[i] = ((accuracy[i]+res))/2;
                System.out.println("\nTesting on " + dataSet1Name + ": " +
                        decimalFormat.format(res) + "%" + " Accuracy using K = " + k);
                k += 2;
            }
            k = initialK;
            System.out.println("\n========== Two Fold Accuracies ==========");
            for (double acc : accuracy)
            {
                System.out.println("\nTwo fold test result for K = "+ k + " is " + decimalFormat.format(acc)+"%");
                k+=2;
            }
        }
        if (RUN_NN){
            System.out.println("\n<<<<<Neural Network>>>>>");
            NeuralNetwork.runNetworkAlgorithm(dataSet1, dataSet1Name, dataSet2, dataSet2Name);
        }

    }

}