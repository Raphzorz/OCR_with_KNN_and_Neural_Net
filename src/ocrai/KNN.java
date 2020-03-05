package ocrai;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class KNN {

    public static Double getDistance(double[][] trainData, double[][] testData, int K) {

        int correct = 0;
        int attempted = 0;
        double distanceBetweenPoints;
        // For every row
        for (double[] datum : testData) {
            List<KNNeighbourList> KNNeighbourList = new ArrayList<>();
            for (double[] trainDatum : trainData) {
                distanceBetweenPoints = 0;
                // For every column
                for (int j = 0; j < 64; j++) {
                    distanceBetweenPoints += Math.pow(trainDatum[j] - datum[j], 2);
                }
                KNNeighbourList.add(new KNNeighbourList(distanceBetweenPoints, trainDatum[64]));
            }
            ++attempted;
            KNNeighbourList.sort(new SortByDistance());
            double[] listOfLabels = new double[K];
            for (int j = 0; j < K; j++) {
                listOfLabels[j] = KNNeighbourList.get(j).label;
            }
            double mostCommonLabel = getMostCommonLabel(listOfLabels);
            if (mostCommonLabel == datum[64])
                ++correct;
        }
        return ((double) correct / attempted * 100);
    }

    public static double getMostCommonLabel(double[] a) {
        int count = 1, tempCount;
        double mostCommon = a[0];
        double temp;
        for (int i = 0; i < (a.length - 1); i++) {
            temp = a[i];
            tempCount = 0;
            for (int j = 1; j < a.length; j++) {
                if (temp == a[j])
                    tempCount++;
            }
            if (tempCount > count) {
                mostCommon = temp;
                count = tempCount;
            }
            if (tempCount == count) {
                // If there is a match in the number of labels then there is a 50% chance of one being chosen over
                // the other
                if (Math.random() >= 5) {
                    mostCommon = temp;
                    count = tempCount;
                }
            }
        }
        return mostCommon;
    }

    static class SortByDistance implements Comparator<KNNeighbourList> {
        @Override
        public int compare(KNNeighbourList a, KNNeighbourList b) {
            return Double.compare(a.distance, b.distance);
        }
    }
}
