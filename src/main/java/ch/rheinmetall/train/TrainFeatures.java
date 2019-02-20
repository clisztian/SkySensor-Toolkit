package ch.rheinmetall.train;

import java.io.IOException;
import java.util.HashMap;

import ch.rheinmetall.data.FeatureExtraction;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class TrainFeatures {

    FeatureExtraction features;
    HashMap<String, Object> params;
    HashMap<String, DMatrix> watches;
	private float trainPercent;
	
     
    public TrainFeatures(String classFile, float trainPercent) throws IOException, XGBoostError {

		this.setTrainPercent(trainPercent);
		features = new FeatureExtraction().loadFeatures(classFile, trainPercent);
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
	    watches.put("train", features.getTrainMatrix());
	    watches.put("test", features.getTestMatrix());		
	}
	
	public void train() throws XGBoostError {
		
	    Booster booster = XGBoost.train(features.getTrainMatrix(), params, 200, watches, null, null);
	    
	    booster.saveModel("resources/data/model_data/features_model.bin");
	    
	    float[][] testPred = booster.predict(features.getTestMatrix(), true);
	    
	    
	    System.out.println(testPred[0][0]);
	    
	}
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		String classFile = "/home/lisztian/eclipse-workspace/0000/features/output.csv";
		TrainFeatures myTrain = new TrainFeatures(classFile, .8f);
		
		
		myTrain.setParameters(0.9, 3, 0, "multi:softmax");
		myTrain.setWatchlist();
		
		myTrain.train();
	}

	public float getTrainPercent() {
		return trainPercent;
	}

	public void setTrainPercent(float trainPercent) {
		this.trainPercent = trainPercent;
	}
	
	
		
}
