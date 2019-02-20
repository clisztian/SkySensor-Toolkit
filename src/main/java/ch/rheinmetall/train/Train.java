package ch.rheinmetall.train;

import java.io.IOException;
import java.util.HashMap;

import ch.rheinmetall.data.DataSets;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class Train {

	public DataSets myData;
	
	HashMap<String, Object> params;
	HashMap<String, DMatrix> watches;
	
	public Train(int imageHeight, int imageWidth, float testPercent) throws IOException, XGBoostError {
	
		myData = new DataSets(testPercent, imageWidth, imageHeight);
		//myData.loadImages();
		//myData.createDataSets();
		myData.createDataSetsFromBinary("resources/data/model_data/train.buffer", 
				"resources/data/model_data/test.buffer");

	}
	
	public void setParameters(double eta, int max_depth, int silent, String objective) {
		
		params = new HashMap<String, Object>();
		params.put("booster", "gbtree");
		params.put("num_class", 25);
	    params.put("eta", eta);
	    params.put("max_depth", max_depth);
	    params.put("silent", silent);
	    params.put("objective", objective);		
	}
	
	public void setWatchlist() {		
	    //specify watchList
	    watches = new HashMap<String, DMatrix>();
	    watches.put("train", myData.getTrainMatrix());
	    watches.put("test", myData.getTestMatrix());		
	}
	
	public void train() throws XGBoostError {
		
	    Booster booster = XGBoost.train(myData.getTrainMatrix(), params, 200, watches, null, null);
	    
	    booster.saveModel("resources/data/model_data/model.bin");
	    
	    float[][] testPred = booster.predict(myData.getTestMatrix(), true);
	    
	    
	    System.out.println(testPred[0][0]);
	    
	}
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		Train myTrain = new Train(64, 64, .1f);
		
		myTrain.setParameters(0.9, 3, 0, "multi:softmax");
		myTrain.setWatchlist();
		
		myTrain.train();
		
	}
}
