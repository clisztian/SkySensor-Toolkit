package ch.rheinmetall.data;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Transparency;
import java.awt.image.BufferedImage;

public class Resize {


		
	public static BufferedImage scale(BufferedImage img, int targetWidth, int targetHeight) {

	    int type = img.getType();
	    BufferedImage ret = img;
	    BufferedImage scratchImage = null;
	    Graphics2D g2 = null;

	    int w = img.getWidth();
	    int h = img.getHeight();

	    int prevW = w;
	    int prevH = h;

	    do {
	        if (w > targetWidth) {
	            w /= 2;
	            w = (w < targetWidth) ? targetWidth : w;
	        }

	        if (h > targetHeight) {
	            h /= 2;
	            h = (h < targetHeight) ? targetHeight : h;
	        }

	        if (scratchImage == null) {
	            scratchImage = new BufferedImage(w, h, type);
	            g2 = scratchImage.createGraphics();
	        }

	        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
	                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
	        g2.drawImage(ret, 0, 0, w, h, 0, 0, prevW, prevH, null);

	        prevW = w;
	        prevH = h;
	        ret = scratchImage;
	    } while (w != targetWidth || h != targetHeight);

	    if (g2 != null) {
	        g2.dispose();
	    }

	    if (targetWidth != ret.getWidth() || targetHeight != ret.getHeight()) {
	        scratchImage = new BufferedImage(targetWidth, targetHeight, type);
	        g2 = scratchImage.createGraphics();
	        g2.drawImage(ret, 0, 0, null);
	        g2.dispose();
	        ret = scratchImage;
	    }

	    return ret;

	}
	
}
