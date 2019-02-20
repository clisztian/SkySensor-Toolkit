package ch.rheinmetall.predict;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import boofcv.gui.ListDisplayPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.image.UtilImageIO;
import ch.rheinmetall.data.DataSets;
import ch.rheinmetall.data.LoadClassInformation;
import ch.rheinmetall.data.Resize;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class Predict {

	
	Booster booster = null;
	private int imageHeight;
	private int imageWidth;
	
	public Predict(String anyModel, int w, int h) throws XGBoostError {
		
		this.loadModel(anyModel);
		this.imageHeight = h;
		this.imageWidth = w;
	}
	
	public void loadModel(String myModel) throws XGBoostError {
		
		booster = XGBoost.loadModel(myModel);	
	}
	
	
	public float[][] predictionPipeline(String file) throws XGBoostError {
		
		BufferedImage out = Resize.scale(UtilImageIO.loadImage(file), imageWidth, imageHeight);
		DataBufferByte dataBuffer = ((DataBufferByte)out.getRaster().getDataBuffer());
		float[] data = new float[imageWidth*imageHeight]; 
		
		int size = dataBuffer.getSize();
		for(int k = 0; k < size; k++) {
			 data[k] = dataBuffer.getElemFloat(k)/255f;
		}
		
		DMatrix anyMatrix = new DMatrix(data, 1, imageWidth*imageHeight);
		
		float[][] predicts = booster.predict(anyMatrix);
		
		
		return predicts;
	}
	
	public void showImage(String myFile) {
		
		File f = new File(myFile);
		
		BufferedImage buffered = UtilImageIO.loadImage(f.getPath());
		if( buffered == null)
			throw new RuntimeException("Couldn't find input image");
		
		DataBufferByte dataBuffer = ((DataBufferByte)buffered.getRaster().getDataBuffer());
		int size = dataBuffer.getSize();
		float[] myImage = new float[size];
		for(int i = 0; i < size; i++) {
			myImage[i] = dataBuffer.getElemFloat(i)/255f;
		}
		
		ListDisplayPanel panel = new ListDisplayPanel();
		
		panel.addImage(buffered,"Original");
		ShowImages.showWindow(panel,"Target",true);
	}
	
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		LoadClassInformation info = new LoadClassInformation();
		info.loadClassInformation("resources/classes.csv"); 
		
		Predict predict = new Predict("resources/data/model_data/model.bin", 64, 64);
		
		String myFile = "/home/lisztian/eclipse-workspace/0000/classification/gradient/test/"; 
		
		//myFile += "0IGVegaCore_000000016.jpg";
		myFile += "12IGVegaCore_000000019.jpg";
		
		predict.showImage(myFile);
		
		float[][] out = predict.predictionPipeline(myFile);
		
		int myClass = (int)out[0][0];
		
		
		System.out.println(info.getClass(myClass));
	}
	
}
