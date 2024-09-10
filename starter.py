import subprocess
from time import time
#from time import time, localtime, strftime, sleep

def run_program(program_name, conda_env,log_file,dataset_name,units,max_time,gpu,tentativo):#,tentativo2):
    try:
        # Avvia il programma specificato all'interno dell'ambiente Conda
        #process = subprocess.run(["conda", "run", "-n", conda_env, "python3", program_name], capture_output=True,timeout=timeout, text=True, check=True)
        print(f'units = {units}')
        print(f'dataset_name = {dataset_name}')
        print(f'str(max_time) = {str(max_time)}')
        print(f'gpu = {gpu}')
        print(f'str(tentativo) = {str(tentativo)}')
        
        
        
        
        process = subprocess.run(["conda", "run", "-n", conda_env, "python", program_name,units,dataset_name,str(max_time),gpu,str(tentativo)#,str(tentativo2)
                        ], capture_output=True, text=True, check=True)

        #print(result.stdout)
        #print(f"Il programma {program_name} è stato eseguito con successo.")
        # Stampa l'output e l'errore standard del programma
        print(f"Output del programma {program_name}:")
        print(process.stdout)
        print(f"Errore standard del programma {program_name}:")
        print(process.stderr)

        # Verifica se il processo è stato terminato correttamente
        if process.returncode == 0:
            print(f"Il programma {program_name} è stato eseguito con successo.")
            return True
        else:
            print(f"Il programma {program_name} ha restituito il codice di uscita {process.returncode}.")
            return False
    except subprocess.TimeoutExpired:
        print(f"Il programma {program_name} ha superato il timeout e viene terminato.")
        return False
    
    except subprocess.CalledProcessError as e:
        # Se il programma genera un errore, cattura il messaggio di errore e lo scrive nel file di log
        error_message = f"Errore nel programma {program_name}:\n\n{e.stderr}\n"
        
        log_file.write(error_message + "\n")
        log_file.flush()
        print(error_message)
        
        return False




if __name__ == "__main__":
    # Lista dei programmi da eseguire in sequenza
    programs_to_run = [
        #'provasleep.py'#
        #"rHybrid.py"
        #'rtestHybrid.py'
        #'Hybrid_altrids.py',
        'Hybrid_altridsTEST.py'
    ]
    # Nome dell'ambiente Conda in cui eseguire i programmi
    conda_env = "tf-hyb"
    max_time = 60*60*24
    gpu = '2'
    #log_file = open("error_log_"+dataset_name+'_'+units+".txt","w")


    
    tentativo = 0
    num_units = [
                '3',
                 '8',
                 '16',
                 #'3'
                ]
    
    dataset_names = [
        #'Trace', 
        #'KeplerLightCurves', #non lo trovo
        
        'Beef', 'Wine',
        'Meat', 'Trace' ,
        #'Wine', 
        #'Beef',
        #'FordA',
        #'FordB',
        #'sMNIST',
        #'psMNIST'
    ]
    for dataset_name in dataset_names:
        for num_unit in num_units:

            while True:
                log_file = open("error_log_"+dataset_name+'_'+num_unit+'_'+str(tentativo)+".txt","w")
                inizio = time()

                terminato = run_program(programs_to_run[0],conda_env,log_file,dataset_name,num_unit,max_time,gpu,tentativo)
                fine = time()
                max_time -=round((fine-inizio)-60*45)
                tentativo+=1
                if terminato:
                    break

    """while True:
        inizio = time()
        terminato = run_program(programs_to_run[0],conda_env,log_file,dataset_name,'8',max_time,gpu,tentativo)
        fine = time()
        max_time -=round((fine-inizio)-60*45)
        tentativo+=1
        if terminato:
            break

    max_time = 60*60*24
    tentativo = 0
    units = '16'
    while True:
        inizio = time()
        terminato = run_program(programs_to_run[0],conda_env,log_file,dataset_name,'16',max_time,gpu,tentativo)
        fine = time()
        max_time -=round((fine-inizio)-60*45)
        tentativo+=1
        if terminato:
            break"""
    
    """dataset_names = [
        'Trace', 
        'KeplerLightCurves', 
        'Meat', 
        'Wine', 
        'Beef',
        'FordA',
        'FordB',
        'sMNIST',
        'psMNIST'
    ]
    num_units = ['3',
                 '8',
                 '16'
                ]
    iniziotutto = time()
    maxtimeNOTTE = 60*60*7
    esci = False

    for dataset_name in dataset_names:
        for num_unit in num_units:
            if dataset_name == 'FordA' and num_unit=='3':
                continue
            if dataset_name == 'FordA' and num_unit=='8':
                continue
            
            for i in range(3):
                max_time = 60*60*24
                tentativo = 0
                if esci:
                    break
                while True:
                    inizio = time()
                    terminato = run_program(programs_to_run[0],conda_env,log_file,dataset_name,num_unit,max_time,gpu,tentativo,i)
                    fine = time()
                    max_time -=round((fine-inizio)-60*45)
                    tentativo+=1
                    if terminato:
                        break
                    if round(time() - iniziotutto) > maxtimeNOTTE:
                        esci = True
                        break"""



    log_file.close()