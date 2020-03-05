package ocrai;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class ReadFile {

    public static double[][] readFile(String filename, boolean NN) {
        // read the data from the file
        ArrayList<String> row = new ArrayList<String>();
        try {
            FileInputStream fileInputStream = new FileInputStream(filename);
            DataInputStream dataInputStream = new DataInputStream(fileInputStream);
            BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream));

            String rowContent;
            while ((rowContent = br.readLine()) != null) {
                row.add(rowContent);
            }
            dataInputStream.close();
        } catch (Exception e) {
            e.printStackTrace(System.out);
        }
        double[][] dataSet = new double[row.size()][65];
        for (int rowNumber = 0; rowNumber < row.size(); rowNumber++) {
            String str = row.get(rowNumber);
            // System.out.println(str);
            String[] allColumns = str.split(",");
            for (int columnNumber = 0; columnNumber < allColumns.length; columnNumber++) {
                dataSet[rowNumber][columnNumber] = Integer.parseInt(allColumns[columnNumber]);
            }
        }
        return dataSet;
    }

}
