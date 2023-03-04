'''
syntax:
python main.py [-h] SCORING_METHOD solutionpath datapath network_model_name model_scenario_number dataset_scenario_number

EXAMPLE: to run scenario1 of 'ds' dataset
Assuming solution1.txt and solution2.txt are in solutions/
python main.py solutions/ ds/ ds_model 1 1

Dependencies in ds/ds_model/:
    inputfiles.ini

inputfiles.ini
.: inputfile is in ds/ds_model
x: inputfile is in ds/ds_model/scenario_<k>  
    [INPUTS]
    ROP=.
    RAW=x
    CON=.
    INL=.
    
'''

import argparse
import csv
import os.path
# <<<<<<< HEAD:py/main.py
import os.path
import sys
import time
import traceback

#=======
#>>>>>>> package:data_utilities/main.py
try:
    import pyopf.preprocess.data_utilities.evaluation as evaluation
except:
    import evaluation


from mpi4py import MPI

comm = MPI.COMM_WORLD
process_count = comm.Get_size()
print('Number of processes: {}'.format(process_count))
process_rank = comm.Get_rank()

print(f'main.py process rank: {process_rank}')


MAXOBJ = 9876543210.0
MAXVIOL = 9876543210.0


def getInputPath(model_path, filetype, scenario_path):

    modelpath_case='%s/%s' % (model_path,filetype)
    print('Looking for %s ' % modelpath_case )

    if os.path.exists( modelpath_case ):
        return model_path
    else:
        return scenario_path


def run():


    # change the arguments as desired
    parser = argparse.ArgumentParser(description='Evaluate a solution to a problem instance')
    
    parser.add_argument('scoring_method', help='scoring method / division: 1/2/3/4')
    parser.add_argument('solutionpath', help='path of the folder containing solution')
    parser.add_argument('datapath', help='path of the folder containing input files')
    parser.add_argument('network_model', help='name of the network model')
    parser.add_argument('model_scenario_number', help='scenario number being evaluated relative to single model')
    parser.add_argument('dataset_scenario_number', help='scenario number being evaluated relative to a multi-model dataset')
    parser.add_argument('zinf_objective', help='zinf objective for empty/infeasible/bad solutions')
    parser.add_argument('code1_runtime', help='code1 runtime')
    parser.add_argument('code2_runtime', help='code2 runtime', action="store", nargs='?')
    parser.add_argument('contingency_count', help='contingency count', action="store", nargs='?')
    parser.add_argument('sec_per_contingency', help='seconds per contingency', action="store", nargs='?')
    parser.add_argument('code2_runtime_goal_sec', help='code2 runtime goal', action="store", nargs='?')
    parser.add_argument('is_sensitive', help='Is the dataset sensitive? yes=1, no=0', action="store", nargs='?')

    #parser.add_argument('out', help='path name of the .csv file to write')
    
    args = parser.parse_args()

    model_path = "{}/{}/".format(args.datapath, args.network_model)
    scenario_path = "%s/scenario_%03d"%(model_path, int(args.model_scenario_number))

    print('Model Path: ', model_path)
    print('Scenario Path: ', scenario_path)

    args_raw = getInputPath(model_path, "case.raw", scenario_path) + "/case.raw"    #"{}/case.raw".format(scenario_path, args.model_scenario_number) 
    args_sup = getInputPath(model_path, "case.json", scenario_path) + "/case.json"
    args_con = getInputPath(model_path, "case.con", scenario_path) + "/case.con"


    #if path.exists(args_raw)==False or path.exists(args_sup)==False or path.exists(args_con)==False:
    #    sys.exit(-1)
    #fi


    #/${NETWORKMODEL}_output${SCENARIO_LOCAL}
    output_path = "{}/{}_output{}".format(args.solutionpath,args.network_model,args.model_scenario_number ) 
    args_sol1 = "{}".format(output_path)
    args_sol2 = "{}/solution2.txt".format(output_path)

    args_summary = "{}/GOCFeasibility.csv".format(output_path)
    args_detail = "{}/{}_DetailedSolution.csv".format(output_path, args.model_scenario_number)

    if (args.scoring_method == '3' or args.scoring_method == '4'):
        line_switching_allowed = True
        xfmr_switching_allowed = True
    elif (args.scoring_method == '1' or args.scoring_method == '2'):
        line_switching_allowed = False
        xfmr_switching_allowed = False
    else:
        line_switching_allowed = None
        xfmr_switching_allowed = None
    #line_switching_allowed = True if args.scoring_method == '3' or args.scoring_method == '4' else None
    #xfmr_switching_allowed = True if args.scoring_method == '3' or args.scoring_method == '4' else None

    obj=MAXOBJ

    try:
        contingency_count=int(args.contingency_count)
    except:
        contingency_count=0

    try:
        sec_per_contingency=float(args.sec_per_contingency)
    except:
        sec_per_contingency="N/A"

    try:
        code2_runtime_goal_sec=float(args.code2_runtime_goal_sec)
    except:
        code2_runtime_goal_sec=999999.0

    print("\tScoring Method:%s"%(args.scoring_method))
    print("\tModel:%s"%(model_path))
    print("\tScenario:%s"%(scenario_path))
    print("\tOutput:%s"%(output_path))
    print("\tRAW:%s"%(args_raw))
    print("\tSUP:%s"%(args_sup))
    print("\tCON:%s"%(args_con))

    #Check if solution1 is valid
    if args.zinf_objective == "0" and args.code1_runtime == "0" and args.code2_runtime == "0":
        try:
            args_summary = "{}/GOCFeasibility_base.csv".format(output_path)
            args_detail = "{}/{}_DetailedSolution_base.csv".format(output_path, args.model_scenario_number)
            
            (obj, infeas, solutions_exist, summary_all_cases) = evaluation.run_main(scenario_path, args_sol1, line_switching_allowed, xfmr_switching_allowed, check_contingencies=False)
            if solutions_exist == False or obj == None:
                raise Exception ("Missing solution or obj is None")

        except:
            traceback.print_exc()
            errfile_path = output_path + '/solution_BASECASE.err'
            with open(errfile_path  , 'w'):
                os.utime(errfile_path, None)
        sys.exit(0)

    #solutions_exist = os.path.isfile(args_sol1) and os.path.isfile(args_sol2)
    missing_solution = 'FALSE' #if solutions_exist else 'TRUE'

    obj=None
    infeas=None

    start_time = time.time()

    zinf_objective = float(args.zinf_objective)
    score = zinf_objective

    code1_runtime = ""
    code2_runtime = ""
    infeasibility_text="N/A"

    try:
        code1_runtime = float(args.code1_runtime) * 1.0
    except:
        code1_runtime = args.code1_runtime

    try:
        code2_runtime = float(args.code2_runtime) * 1.0
    except:
        code2_runtime = args.code2_runtime
    

    solutions_exist = False
    try:

        #if solutions_exist:
        (obj, infeas, solutions_exist,summary_all_cases) = evaluation.run_main(scenario_path, args_sol1, line_switching_allowed, xfmr_switching_allowed, check_contingencies=True)


        if process_rank == 0 and solutions_exist != False:

            if obj > zinf_objective and infeas == 0:   #larger than zinf and feasible
                print("obj > infeasible_score and infeas == 0")
                score = obj
            #elif abs(abs(zinf_objective) - MAXOBJ) < 1:       #zinf objective is not available, capture worst case score
            #    print("infeasible score not available")
            #    score = obj
            #    infeas = 1
            elif obj < zinf_objective: #and infeas == 0:   #larger than zinf and feasible
                print("obj < zinf_objective and infeas == 0")
                score = obj
                #infeas = 1
            elif obj == float('nan'):
                score = zinf_objective
            elif infeas == 1:
                score = zinf_objective

            eval_runtime = time.time() - start_time

            infeasibility_text = 'TRUE' if (infeas==1) else 'FALSE' if infeas==0 else 'N/A'


            with open(args_summary, 'w') as summaryfile:
                csvwriter = csv.writer(summaryfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['Scenario','Score', 'Objective',  'Infeasibility','Bad/Missing Solution','Zinf Objective', 'Evaluation Duration (sec)', 'Code 1 Duration (sec)', 'Code 2 Duration (sec)','Contingency Count', 'Seconds per Contingency','Code 2 Runtime Goal (sec)'])
                csvwriter.writerow(['{} Scenario {} (output{})'.format(args.network_model, args.model_scenario_number,args.model_scenario_number), score, obj,  infeasibility_text, missing_solution, zinf_objective,eval_runtime, code1_runtime, code2_runtime, contingency_count,sec_per_contingency,code2_runtime_goal_sec ])

            print("\tSolutions generated:{}".format(solutions_exist))
             
            '''
            try:
                if args.is_sensitive == "1":
                    obj -= zinf_objective
                    score -= zinf_objective
            except:
                pass

            if args.is_sensitive == "1":
                contingency_count = 0
                code2_runtime_goal_sec = 0
                zinf_objective = 0
            '''


            if solutions_exist:
                print("\tModel:{}".format(args.network_model))
                print("\tInfeasible Score:%f"%(zinf_objective))
                print("\tObjective:%f"%(obj))
                print("\tInfeasibility:%d"%(infeas))
                print("\tEval runtime:%f"%(eval_runtime))
                print("\tCode 1 runtime:{}".format(code1_runtime))
                print("\tCode 2 runtime:{}".format(code2_runtime))
                print("\tsec_per_contingency:{}".format(sec_per_contingency))
                print("\tScore:%f"%(score))
                print("\tContingency Count:%d"%(contingency_count))
                print("\tcode2_runtime_goal_sec:%f"%(code2_runtime_goal_sec))


    except Exception as e:
        traceback.print_exc()
        print(e)
        missing_solution = 'TRUE' if not solutions_exist else 'FALSE'
        with open(args_summary, 'w') as summaryfile:
            csvwriter = csv.writer(summaryfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['Scenario','Score', 'Objective',  'Infeasibility','Bad/Missing Solution','Zinf Objective', 'Evaluation Duration (sec)', 'Code 1 Duration (sec)', 'Code 2 Duration (sec)','Contingency Count','Seconds per Contingency','Code 2 Runtime Goal (sec)'])
            csvwriter.writerow(['{} Scenario {} (output{})'.format(args.network_model, args.model_scenario_number,args.model_scenario_number), score, 'N/A',  'N/A', missing_solution, zinf_objective,'', code1_runtime, code2_runtime, contingency_count, 'N/A', code2_runtime_goal_sec])
    
    
if __name__ == '__main__':
    run()
