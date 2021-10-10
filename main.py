from simulate.simulation import MonteCarlo

if __name__ == "__main__":
    simulation = MonteCarlo(20)
    simulation.TEMPERATURE = 400
    simulation.TOTCONF = 5000
    simulation.flag = "DEBUG"
    simulation.comments = "Test static methods"
    simulation.run()
