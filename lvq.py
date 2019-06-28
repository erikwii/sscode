class lvq:
    def __init__(self, num_class, training_data = null, test_data = null):
        self.numClass = num_class
        self.weight = []
        self.trainingData = []
        self.testData = []

    def setTrainingData(self, training_data):
        self.trainingData = training_data
    
    def getTrainingData(self):
        return self.trainingData

    def setTestData(self, test_data):
        self.testData = test_data
    
    def getTestdata(self):
        return self.testData