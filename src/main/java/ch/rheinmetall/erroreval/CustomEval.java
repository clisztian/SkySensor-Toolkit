package ch.rheinmetall.erroreval;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.IEvaluation;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class CustomEval implements IEvaluation {
	  /**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private static final Log logger = LogFactory.getLog(CustomEval.class);

	  String evalMetric = "custom_error";

	  @Override
	  public String getMetric() {
	    return evalMetric;
	  }

	  @Override
	  public float eval(float[][] predicts, DMatrix dmat) {
	    float error = 0f;
	    float[] labels;
	    try {
	      labels = dmat.getLabel();
	    } catch (XGBoostError ex) {
	      logger.error(ex);
	      return -1f;
	    }
	    int nrow = predicts.length;
	    for (int i = 0; i < nrow; i++) {
	      if (labels[i] == 0f && predicts[i][0] > 0.5) {
	        error++;
	      } else if (labels[i] == 1f && predicts[i][0] <= 0.5) {
	        error++;
	      }
	    }

	    return error / labels.length;
	  }

	}