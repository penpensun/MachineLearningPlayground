package org.peng.ml.regression.LinearRegressioJava;
import java.io.*;
import java.util.*;
/**
 * This class implements the linear regression using gradient descent.
 * @author penpen926
 *
 */
public class GradientDescentLinearRegression {
	
	public static float[] gradientDescent(float[][] indicatorData, float[] observedData, int iter,float alpha) {
		float[] parameters = new float[indicatorData[0].length];
		// Init the parameters
		for(int i=0;i<parameters.length;i++) 
			parameters[i] = 0;
		//Create matrix object for indicatorData and observedData
		Matrix indicatorMatrix = new Matrix(indicatorData);
		Matrix obervedMatrix = new Matrix(observedData,VectorType.COL_VEC);
		Matrix parameterMatrix = new Matrix(parameters, VectorType.COL_VEC);
		
		// For a certain number of iterations, compute the partial differentiates and the gradient descent.
		for(int i=0;i<iter;i++) {
			Matrix error = Matrix.subtract(Matrix.multiply(indicatorMatrix, parameterMatrix),
					obervedMatrix);
			Matrix differentiates = Matrix.multiply(indicatorMatrix.getTranspose(), error);
			parameterMatrix = Matrix.subtract(parameterMatrix, Matrix.multiply(alpha/indicatorData.length, differentiates));
		}
		
		for(int i=0;i<parameters.length;i++) 
			parameters[i] = parameterMatrix.getElement(i, 0);
		return parameters;
	}
	
	
	public float[][] readin(String file){
		FileReader fr = null;
		BufferedReader br = null;
		try {
			fr = new FileReader(file);
			br = new BufferedReader(fr);
		}catch(IOException e) {
			e.printStackTrace();
		}
		ArrayList<float[]> arrayListData = new ArrayList<>();
		String line = null;
		try {
			while((line = br.readLine())!= null) {
				String[] split = line.split(",");
				float[] data = new float[split.length];
				for(int i=0;i<data.length;i++)
					data[i] = Float.parseFloat(split[i]);
				arrayListData.add(data);
			}
		}catch(IOException e) {
			e.printStackTrace();
		}
		arrayListData.trimToSize();
		float[][] ans = new float[arrayListData.size()][arrayListData.get(0).length];
		arrayListData.toArray(ans);
		return ans;
	}
	
	public static void run() {
		String inputFile = "../../data/regression_data1.txt";
		GradientDescentLinearRegression gdlr = new GradientDescentLinearRegression();
		float[][] completeData = gdlr.readin(inputFile);
		float[][] indicatorData= new float[completeData.length][completeData[0].length];
		float[] observedData = new float[completeData.length];
		
		for(int i=0;i<completeData.length;i++) {
			for( int j=0;j<completeData[0].length-1;j++)
				indicatorData[i][j] = completeData[i][j];
			observedData[i] = completeData[i][completeData[0].length-1];
			indicatorData[i][completeData[0].length-1] = 1.0f;
		}
		int iter =1000;
		float alpha = 0.01f;
		float[] parameters = GradientDescentLinearRegression.gradientDescent(indicatorData, observedData, iter, alpha);
		/*
		for(int i=0;i<parameters.length;i++)
			System.out.print(parameters[i]+"  ");
		System.out.println();
		*/
	}
	
	
	public static void main(String args[]) {
		long beforeRun = System.currentTimeMillis();
		for(int i=0;i<100;i++)
			run();
		long afterRun = System.currentTimeMillis();
		System.out.println((afterRun - beforeRun)/1000.0);
	}
}