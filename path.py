from project import stacking as st
import deepnetwork as dp
#from  project import deepnetwork as pd


#namespace = ['antivirus','bach_choral']

#['indian_liver_patient','lsvt','mlprove','online_news_popularity','parkinson2','planning_relax','student_alcohol','susy','wilt']
'''''
namespace = ['antivirus','bach_choral','banknote','breast_cancer','chronic_kidney','climate','cnae9','default_cc',
             'diabetic_retinopathy','eeg_eye_state','fertility','forest','gas2','gesture','grammatical_facial',
             'heart','hepmass','hill_valley','indian_liver_patient','insurance-company-coil2000','ionosphere']
'''''

#'''''
namespace = ['antivirus','bach_choral','banknote','breast_cancer','chronic_kidney','climate','cnae9','default_cc',
             'diabetic_retinopathy','eeg_eye_state','fertility','forest','gas2','gesture','grammatical_facial',
             'heart','hepmass','hill_valley','indian_liver_patient','insurance-company-coil2000','ionosphere',
             'isolet','libras','lsvt','mfeat','mhealth','micromass','mlprove','musk','occupancy','online_news_popularity',
             'ozone','parkinson2','parkinsons','phishing_websites','planning_relax','qsar_biodeg','secom',
             'seeds','seismic','smartphone','sonar','spambase','steel_faults','student_alcohol','thoraric',
             'tv_news_channel','urban_land','vertebral','wall_follow','wilt','susy']
#'''
print(len(namespace))
for i in namespace:
    st.result.write(i + ",")
    for j in range(10):
        path1 = "/home/kalyani/Desktop/datasets_v1/"+i+"/data"
        path2 = "/home/kalyani/Desktop/datasets_v1/"+i+"/random_class."+str(j)
        path3 = "/home/kalyani/Desktop/datasets_v1/"+i+"/trueclass"
        print(path1,path2,path3,i)
        st.main(path1,path2,path3,i)

        #dp.main(path1, path2, path3, i)

st.result.close()
#dp.result.close()
