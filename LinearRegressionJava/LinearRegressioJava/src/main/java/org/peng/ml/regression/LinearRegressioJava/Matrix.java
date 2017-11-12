package org.peng.ml.regression.LinearRegressioJava;

public class Matrix {
	float[][] elements;
	public Matrix() {
		elements = null;
	}
	
	public Matrix(float[][] elements) {
		this.elements = elements;
	}
	
	public Matrix(int rowNo, int colNo) {
		elements = new float[rowNo][colNo];
		for(int i=0;i<rowNo;i++)
			for(int j=0;j<colNo;j++)
				elements[i][j] =0;
	}
	
	public float[][] getElements(){
		return elements;
	}
	
	public float getElement(int i, int j) {
		return elements[i][j];
	}
	
	public boolean isEmpty() {
		return (elements == null);
	}
	
	public int getRowNum() {
		if(elements == null)
			return 0;
		else
			return elements.length;
	}
	
	public int getColNum() {
		if(elements == null)
			return 0;
		else return elements[0].length;
	}
	
	public void setElement(int i, int j, float e) {
		elements[i][j] = e;
	}
	
	public static Matrix multiply(Matrix m1, Matrix m2) {
		//check if two matrix could be multiplied
		if(m1.getColNum() != m2.getRowNum()
				|| m1.isEmpty() || m2.isEmpty())
			throw new IllegalArgumentException("m1 is null or m2 is null, or the column no. of m1 is not equal to row no. of m2.");
		//Create the new matrix
		Matrix ans = new Matrix(m1.getRowNum(), m2.getColNum());
		for(int i=0;i<ans.getRowNum();i++)
			for(int j=0;j<ans.getColNum();j++) {
				// the element i,j of ans is the inner product of
				// vector m1 row i and vector m2 column j;
				for(int k = 0;k<m1.getColNum();k++) {
					//System.out.println(m2.getColNum()+"  "+m2.getRowNum());
					//System.out.println("m1 column number:  "+m1.getColNum());
					//System.out.println("k  :"+k+"  j:  "+j);
					//System.out.println(m2.getElement(k, j));
					ans.setElement(i,j, 
							ans.getElement(i, j)+ (m1.getElement(i, k) * m2.getElement(k, j)));
				}
			}
		
		return ans;
	}
	
	public void printMatrix() {
		for(int i=0;i<elements.length;i++) {
			for(int j=0;j<elements[0].length;j++)
				System.out.print(elements[i][j]+"  ");
			System.out.println();
		}
	}
	
	
	
	public static void main(String args[]) {
		//Matrix 1;
		float e1[][] = {{1.0f,2.0f,3.0f}};
		float e2[][] = {{1.0f},{2.0f},{3.0f}};
		Matrix m1 = new Matrix(e1);
		Matrix m2 = new Matrix(e2);
		Matrix ans = Matrix.multiply(m2, m1);
		ans.printMatrix();
		
	}
	
}
