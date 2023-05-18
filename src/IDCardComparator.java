import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class IDCardComparator {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Load the two ID card images
        Mat idCard1 = Imgcodecs.imread("images/sh.jpg");
        Mat idCard2 = Imgcodecs.imread("images/2.jpg");

        // Convert the images to grayscale
        Mat gray1 = new Mat();
        Mat gray2 = new Mat();
        Imgproc.cvtColor(idCard1, gray1, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(idCard2, gray2, Imgproc.COLOR_BGR2GRAY);

        // Detect keypoints and compute descriptors using ORB
        ORB orb = ORB.create();
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        orb.detectAndCompute(gray1, new Mat(), keypoints1, descriptors1);
        orb.detectAndCompute(gray2, new Mat(), keypoints2, descriptors2);

        // Match the descriptors using brute-force matching
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors1, descriptors2, matches);

        // Filter matches based on a distance threshold (you can adjust this value)
        double threshold = 50.0;
        List<DMatch> goodMatches = new ArrayList<>();
        for (DMatch match : matches.toArray()) {
            if (match.distance <= threshold) {
                goodMatches.add(match);
            }
        }

        // Calculate a similarity score based on the number of good matches
        double similarityScore = (double) goodMatches.size() / (double) keypoints1.toList().size();

        // Print the similarity score
        System.out.println("Similarity Score: " + similarityScore);
    }
}


