package is.ispit;

import org.neuroph.core.NeuralNetwork;

public class Training{
    private NeuralNetwork neuralNet;
    private double mse;

    public Training(NeuralNetwork neuralNet, double mse) {
        this.neuralNet = neuralNet;
        this.mse = mse;
    }

    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    public double getMse() {
        return mse;
    }

    public void setNeuralNet(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }

    public void setMse(double mse) {
        this.mse = mse;
    }
    
}