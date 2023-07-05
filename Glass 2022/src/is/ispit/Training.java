package is.ispit;

import org.neuroph.core.NeuralNetwork;

public class Training{
    private NeuralNetwork neuralNet;
    private double accuraccy;

    public Training(NeuralNetwork neuralNet, double accuraccy) {
        this.neuralNet = neuralNet;
        this.accuraccy = accuraccy;
    }

    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    public double getAccuraccy() {
        return accuraccy;
    }

    public void setNeuralNet(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }

    public void setAccuraccy(double accuraccy) {
        this.accuraccy = accuraccy;
    }
    
}