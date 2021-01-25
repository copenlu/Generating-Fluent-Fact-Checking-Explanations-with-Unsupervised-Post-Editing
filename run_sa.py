import json
import math

from editor import RobertaEditor

if __name__=="__main__":


    editor  = RobertaEditor()
    editor.cuda()
    
    simulated_annealing = SimulatedAnnealing(editor)    