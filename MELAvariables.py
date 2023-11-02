from __future__ import division, print_function
from math import acos
from ROOT import TLorentzVector, TVector3
import numpy as np
import pandas as pd
import time
import math

'''
def computeDerivedMELAVariables(variablesToDerive, dataFrame, signal, analysis):

    newColumns = getMELAvarFromComponents(dataFrame['lep1_pt']  , dataFrame['lep1_eta']  , dataFrame['lep1_phi']  , 
                                          dataFrame['lep2_pt']  , dataFrame['lep2_eta']  , dataFrame['lep2_phi']  , 
                                          dataFrame['sigVJ1_pt'], dataFrame['sigVJ1_eta'], dataFrame['sigVJ1_phi'],
                                          dataFrame['sigVJ2_pt'], dataFrame['sigVJ2_eta'], dataFrame['sigVJ2_phi'] )
                                              

    index =0
    for variableToDerive in variablesToDerive:
        print(variableToDerive)
        dataFrame[variableToDerive] = newColumns[index]
        index++

    return dataFrame
'''


def computeDerivedMELAVariables(MELAvarNames, df): # starting fom the dataframe
                                          

    #dfmela = pd.DataFrame(columns=MELAvarNames)
    dfmela = pd.DataFrame(np.random.randn(len(df), len(MELAvarNames)),columns=MELAvarNames)

    partonMass = 0.
    vl1 = TLorentzVector()
    vl2 = TLorentzVector()
    vj1 = TLorentzVector()
    vj2 = TLorentzVector()
    curr_time = time.time()
    start_time= curr_time
    print("computeDerivedMELAVariables:: starting time is ",curr_time)

    for i in range(len(df)):

        dl1pt  =np.double( df.loc[i, "lep1_pt"    ] );
        dl1eta =np.double( df.loc[i, "lep1_eta"   ] );
        dl1phi =np.double( df.loc[i, "lep1_phi"   ] );

        dl2pt  =np.double( df.loc[i, "lep2_pt"    ] );
        dl2eta =np.double( df.loc[i, "lep2_eta"   ] );
        dl2phi =np.double( df.loc[i, "lep2_phi"   ] );
        
        dj1pt  =np.double( df.loc[i, "sigVJ1_pt"  ] );
        dj1eta =np.double( df.loc[i, "sigVJ1_eta" ] );
        dj1phi =np.double( df.loc[i, "sigVJ1_phi" ] );

        dj2pt  =np.double( df.loc[i, "sigVJ2_pt"  ] );
        dj2eta =np.double( df.loc[i, "sigVJ2_eta" ] );
        dj2phi =np.double( df.loc[i, "sigVJ2_phi" ] );

        vl1.SetPtEtaPhiM (dl1pt, dl1eta, dl1phi, np.double(partonMass))
        vl2.SetPtEtaPhiM (dl2pt, dl2eta, dl2phi, np.double(partonMass))
        vj1.SetPtEtaPhiM (dj1pt, dj1eta, dj1phi, np.double(partonMass))
        vj2.SetPtEtaPhiM (dj2pt, dj2eta, dj2phi, np.double(partonMass))

        cthstr, phi, phi1, cth1, cth2, X_Y, X_Pt = getMELAvar(vl1, vl2, vj1, vj2)
        #dataframe.at[index,'column-name']='new value'
        dfmela.loc[i] = [cthstr, phi, phi1, cth1, cth2, X_Y, X_Pt]
        if i%100000 == 0:
            print("index i="+str(i)+" mela var = "+str(cthstr)+" "+str(phi)+" "+str(phi1)+" "+str(cth1)+" "+str(cth2)+" "+str(X_Y)+" "+str(X_Pt))
            newtime = time.time()
            dtime = newtime-curr_time
            print("elapsed time (for 100000 events)= ",dtime)
            curr_time=newtime


    DT=(curr_time-start_time)/60. 
    print('computeDerivedMELAVariables: variables computed for ',i,' events in ',DT, ' minutes')
    df_concat = pd.concat([df, dfmela], axis=1)
    print('computeDerivedMELAVariables: input and new variables in a single concatenated DF ')
        
    return df_concat
    
'''
def getMELAvarFromComponents(l1pt, l1eta, l1phi, 
                             l2pt, l2eta, l2phi,
                             j1pt, j1eta, j1phi,
                             j2pt, j2eta, j2phi  ): # arguments are: pt eta phi, for l1,l2,j1,j2 
                                          

    partonMass = 0.
    vl1 = TLorentzVector()
    dl1pt  =np.double(l1pt) ;
    dl1eta =np.double(l1eta);
    dl1phi =np.double(l1phi);
    vl1.SetPtEtaPhiM (dl1pt, dl1eta, dl1phi, np.double(partonMass))
    vl2 = TLorentzVector()
    vl2.SetPtEtaPhiM (double(l2pt), double(l2eta), double(l2phi), double(partonMass))
    vj1 = TLorentzVector()
    vj1.SetPtEtaPhiM (double(j1pt), double(j1eta), double(j1phi), double(partonMass))
    vj2 = TLorentzVector()
    vj2.SetPtEtaPhiM (double(j2pt), double(j2eta), double(j2phi), double(partonMass))

    #cthstr, phi, phi1, cth1, cth2, X_Y, X_Pt = getMELAvar(v1, v2, v3, v4)
    #return cthstr, phi, phi1, cth1, cth2, X_Y, X_Pt
    return getMELAvar(vl1, vl2, vj1, vj2)
   
''' 
def getMELAvar( vl1, vl2, vj1, vj2 ): # arguments are TLorentzVector
    
    # TLorentzVector of leptonic V 
    Vl = TLorentzVector(vl1+vl2)
    
    # TLorentzVector of hadronic V 
    Vh = TLorentzVector(vj1+vj2)

    # TLorentzVector of X
    X = TLorentzVector(Vl + Vh)


    #print ("Zl components are: ", Zl.Px(), " ", Zl.Py(), " " , Zl.Pz(), " ", Zl.E())
    #print ("Zh components are: ", Zh.Px(), " ", Zh.Py(), " " , Zh.Pz(), " ", Zh.E())
    #print ("X  components are: ", X.Px() , " ",  X.Py(), " " ,  X.Pz(), " ",  X.E())

    #RFR means reference frame :)
    #definitions from http://arxiv.org/pdf/1208.4018.pdf page 3 [VI]
	
    Vl.Boost( -( X.BoostVector() ) ) # go to X RFR
    Vh.Boost( -( X.BoostVector() ) )

    vl = TVector3(Vl.Vect().Unit())
    vh = TVector3(Vh.Vect().Unit())

    #Costh*
    cthstr = vl.Z();

    #Boost the leptons into the Higgs rest frame:
    vv1 = vl1
    vv2 = vl2 
    vv3 = vj1 
    vv4 = vj2     
    vv1.Boost( -( X.BoostVector() ) ) #go to Higgs RFR
    vv2.Boost( -( X.BoostVector() ) )
    vv3.Boost( -( X.BoostVector() ) )
    vv4.Boost( -( X.BoostVector() ) )
    
    #print ("vv1 components are: ", vv1.Px(), " ", vv1.Py(), " ", vv1.Pz(), " ", vv1.E())
    #print ("vv2 components are: ", vv2.Px(), " ", vv2.Py(), " ", vv2.Pz(), " ", vv2.E())
    #print ("vv3 components are: ", vv3.Px(), " ", vv3.Py(), " ", vv3.Pz(), " ", vv3.E())
    #print ("vv4 components are: ", vv4.Px(), " ", vv3.Py(), " ", vv4.Pz(), " ", vv4.E())
    
    v1p = TVector3(vv1.Vect())
    v2p = TVector3(vv2.Vect())
    v3p = TVector3(vv3.Vect())
    v4p = TVector3(vv4.Vect())
    nz  = TVector3(0,0,1)
    
    #print ("vp1 components are: ", v1p.Px(), " ", v1p.Py(), " ", v1p.Pz())
    #print ("vp2 components are: ", v2p.Px(), " ", v2p.Py(), " ", v2p.Pz())
    #print ("vp3 components are: ", v3p.Px(), " ", v3p.Py(), " ", v3p.Pz())
    #print ("vp4 components are: ", v4p.Px(), " ", v4p.Py(), " ", v4p.Pz())
    #print ("nz  components are: ", nz.Px() , " ", nz.Py() , " ", nz.Pz())

    # Create dot for the vectors
    n1p  = v1p.Cross(v2p).Unit() #prodotto vettoriale tra p1 e p2
    n2p  = v3p.Cross(v4p).Unit() #prodotto vettoriale tra p2 e p4
    nscp =  nz.Cross(vl).Unit()  #prodotto vettoriale tra nz e z1

    #print ("n1p components are: ", n1p.Px(), "", n1p.Py(), n1p.Pz())
    #print ("n2p components are: ", n2p.Px(), "", n2p.Py(), n2p.Pz())
    #print ("nscp components are: ", nscp.Px(), "", nscp.Py(), nscp.Pz())
    #print ('argomento acos(phi): ' +str( -n1p.Dot( n2p ) ))
    #print ('argomento acos(phi1):' +str(n1p.Dot( nscp )))
    

    #phi, phi1
    phi  = ( vl.Dot( n1p.Cross( n2p ) )  / abs( vl.Dot( n1p.Cross( n2p  ) ) ) * acos( -n1p.Dot( n2p ) ) )
    phi1 = ( vl.Dot( n1p.Cross( nscp ) ) / abs( vl.Dot( n1p.Cross( nscp ) ) ) * acos( n1p.Dot( nscp ) ) )

    if phi>math.pi:
        phi = phi - 2.*math.pi
    if phi<math.pi:
        phi = phi + 2.*math.pi
        
    if phi1>math.pi:
        phi1 = phi1 - 2.*math.pi
    if phi1<math.pi:
        phi1 = phi1 + 2.*math.pi

    #print("the first angle  Phi  is :" +str(phi))
    #print("the second angle Phi1 is :" +str(phi1))


    #Costh1,2
    Vh_rfr_Vl = TLorentzVector(Vh)           #it's still in H RFR
    Vh_rfr_Vl.Boost( -( Vl.BoostVector() ) ) #now it's in Z1 RFR (both Z1 and Z2 are in H RFR)
    vh_rfr_Vl = TVector3(Vh_rfr_Vl.Vect())
    
    Vl_rfr_Vh = TLorentzVector(Vl)           #it's still in H RFR
    Vl_rfr_Vh.Boost( -( Vh.BoostVector() ) ) #now it's in Z2 RFR (both Z1 and Z2 are still in H RFR)
    vl_rfr_Vh = TVector3(Vl_rfr_Vh.Vect())

    vv1.Boost( -( Vl.BoostVector() ) ) #Z1 and Z2 still in H RFR: put leptons in their Z's reference frame
    vv3.Boost( -( Vh.BoostVector() ) ) 
    
    cth1 = - ( vh_rfr_Vl.Dot( vv1.Vect() ) / abs( vh_rfr_Vl.Mag() * vv1.Vect().Mag() ) ) #cosenotheta1
    cth2 = - ( vl_rfr_Vh.Dot( vv3.Vect() ) / abs( vl_rfr_Vh.Mag() * vv3.Vect().Mag() ) ) #cosenotheta2
    #print ("the first cosine for theta1 is: " +str(cth1))
    #print ("the first cosine for theta2 is: " +str(cth2))

    #m_VV = H.Mag() #X_ZZ_resolved_m 
    X_Y  = X.Rapidity()
    X_Pt = X.Pt()

    #print ("the rapidity of the sistem VV is:" +str(Y))
    #print ("the transverse momentum of the sistem VV is:" +str(Pt))

    return cthstr, phi, phi1, cth1, cth2, X_Y, X_Pt

