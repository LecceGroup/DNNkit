#include "chains/chainSet_33_22_pdnn.h"
#include "initChains_33_22_pdnn.C"
#include <fstream>
#include <iostream>

void makeMyQuickPlot2022_pdnn(std::string myvar, int nBins=200, double xmin=0., double xmax=100.,
			      bool logx=false, bool logy=false, chainSet_33_22_pdnn* cs=NULL, 
			      std::string userCut="", std::string tag="", bool useW=true, bool shapeComp=true,
			      std::string scoreSignal="Radion", bool savePlots=false)
{
  bool showSignal = false;
  bool showData = false;
  
  if (userCut.find("ZCR") != std::string::npos) {
    showData = true;
  }
  else
    {
      //showSignal = true;
      showData = true;
    }

  std::string anStr = "resolved";
  if (userCut.find("Merg") != std::string::npos) {
    anStr = "merged";
  }
  std::string chStr = "ggF";
  if (userCut.find("VBF") != std::string::npos) {
    chStr = "VBF";
  }
  // get pointers to the chains ....
  if (cs==NULL) {std::cout<<"makeMyQuickPlot2022_pdnn::NULL pointer to chains-set"<<std::endl;return;}
  initChains_33_22_pdnn(scoreSignal, anStr, chStr, cs);
  TChain* fzjet  =cs->_fzjet  ;
  TChain* fdb    =cs->_fDB    ; 
  TChain* fwjet  =cs->_fwjet  ;
  TChain* fttbar =cs->_fttbar ; 
  TChain* fstop  =cs->_fstop  ; 
  TChain* fdata  =cs->_fdata  ; 
  TChain* fSignal=cs->_fsignal;
  std::cout<<"Running for signal/analysis/channel = "<<scoreSignal<<"/"<<anStr<<"/"<<chStr<<std::endl;
  std::cout<<"pointer to cs->_fzjet = <"<<cs->_fzjet<<" >     fzjet=<"<<fzjet<<">"<<std::endl;
  std::cout<<"Variable, userCut = "<<myvar<<" "<<userCut<<std::endl;
	

  /*
  std::string myCutSCR="Pass_MergHP_GGF_ZZ_Tag_SR == 1 || Pass_MergHP_GGF_ZZ_Untag_SR == 1 || Pass_MergHP_GGF_WZ_SR == 1 || Pass_MergLP_GGF_ZZ_Tag_SR == 1 || Pass_MergLP_GGF_ZZ_Untag_SR == 1 || Pass_MergHP_GGF_ZZ_Tag_ZCR == 1 || Pass_MergHP_GGF_WZ_ZCR == 1 || Pass_MergHP_GGF_ZZ_Untag_ZCR == 1 || Pass_MergLP_GGF_ZZ_Tag_ZCR == 1 || Pass_MergLP_GGF_ZZ_Untag_ZCR == 1 || Pass_MergLP_GGF_WZ_SR == 1 || Pass_MergLP_GGF_WZ_ZCR == 1";

  std::string myCutSR="Pass_MergHP_GGF_ZZ_Tag_SR == 1 || Pass_MergHP_GGF_ZZ_Untag_SR == 1 || Pass_MergHP_GGF_WZ_SR == 1 || Pass_MergLP_GGF_ZZ_Tag_SR == 1 || Pass_MergLP_GGF_ZZ_Untag_SR == 1";
  */

  std::string doNotPassAnyMergedGGF = "(!(Pass_MergHP_GGF_ZZ_Tag_SR==1 || Pass_MergHP_GGF_ZZ_Untag_SR==1 || Pass_MergHP_GGF_WZ_SR==1 || Pass_MergHP_GGF_ZZ_Tag_ZCR==1|| Pass_MergHP_GGF_ZZ_Untag_ZCR==1|| Pass_MergHP_GGF_WZ_ZCR==1||Pass_MergLP_GGF_ZZ_Tag_SR==1 || Pass_MergLP_GGF_ZZ_Untag_SR==1 || Pass_MergLP_GGF_WZ_SR==1 || Pass_MergLP_GGF_ZZ_Tag_ZCR==1|| Pass_MergLP_GGF_ZZ_Untag_ZCR==1|| Pass_MergLP_GGF_WZ_ZCR==1))";

  std::string doNotPassAnyMergedVBF = "(!(Pass_MergHP_VBF_ZZ_SR==1 || Pass_MergHP_VBF_ZZ_ZCR==1 || Pass_MergHP_VBF_WZ_SR==1 || Pass_MergHP_VBF_WZ_ZCR==1 || Pass_MergLP_VBF_ZZ_SR==1 || Pass_MergLP_VBF_ZZ_ZCR==1 || Pass_MergLP_VBF_WZ_SR==1 || Pass_MergLP_VBF_WZ_ZCR==1))";

  std::string doNotPassSRMergedGGF = "(!(Pass_MergHP_GGF_ZZ_Tag_SR==1 || Pass_MergHP_GGF_ZZ_Untag_SR==1 || Pass_MergHP_GGF_WZ_SR==1 || Pass_MergLP_GGF_ZZ_Tag_SR==1 || Pass_MergLP_GGF_ZZ_Untag_SR==1 || Pass_MergLP_GGF_WZ_SR==1))";  
  std::string doNotPassSRMergedVBF = "(!(Pass_MergHP_VBF_ZZ_SR==1 ||  Pass_MergHP_VBF_WZ_SR==1 ||  Pass_MergLP_VBF_ZZ_SR==1 || Pass_MergLP_VBF_WZ_SR==1))";

  std::string doNotPassSRResGGF = "(!(Pass_Res_GGF_ZZ_Tag_SR==1 || Pass_Res_GGF_ZZ_Untag_SR==1 || Pass_Res_GGF_WZ_SR==1))";
  std::string doNotPassSRResVBF = "(!(Pass_Res_VBF_ZZ_SR==1 ||  Pass_Res_VBF_WZ_SR==1))";

  std::string doNotPassSRRes =    "("+doNotPassSRResGGF     +"&&"+ doNotPassSRResVBF    +")"; 
  std::string doNotPassMerg  =    "("+doNotPassAnyMergedGGF +"&&"+ doNotPassAnyMergedVBF+")";

  gStyle->SetOptStat(0);
  std::string myCut  = "";//myCutSCR;
  std::string myCutS = "";//myCut;

  std::string lunga = userCut;
  std::string toErase="Pass_";
  size_t pos = lunga.find(toErase);
  if (pos != std::string::npos) lunga.erase(pos, toErase.length());
  toErase="==1";
  pos = lunga.find(toErase);
  if (pos != std::string::npos) lunga.erase(pos, toErase.length());
  toErase="==0";
  pos = lunga.find(toErase);
  if (pos != std::string::npos) lunga.erase(pos, toErase.length());
  std::string tagSel=tag+"_"+lunga;

  bool orthogonal = false;
  if (orthogonal) {
    if (userCut.substr(0,12)=="Pass_Res_GGF") {
      userCut = userCut+"&&"+doNotPassSRMergedGGF;
      //std::cout<<"Redefining userCut as <"<<userCut<<">"<<std::endl;
    }
    else
      {
	//std::cout<<"usercut == <"<<userCut<<">"<<std::endl;
      }
    if (userCut.substr(0,12)=="Pass_Res_VBF") userCut = userCut+"&&"+doNotPassSRMergedVBF;
    
    if (userCut.substr(0,9)=="Pass_Merg" && userCut.find("ZCR")!=std::string::npos ) {
      userCut = userCut+"&&"+doNotPassSRRes;
    }
    
    if (userCut.substr(0,8)=="Pass_Res" && userCut.find("ZCR")!=std::string::npos ) {
      userCut = userCut+"&&"+doNotPassMerg;
    }
  }

  if (userCut!="" && myCut!="")
    {
      myCut = myCut+"&&"+userCut;
    }
  else if (userCut!="")myCut = userCut;



  if (useW) {
    myCut="weight*("+myCut+")";
    myCutS="1000.*weight*("+myCut+")";
  }
  std::cout<<"\nCut is = "<<myCut<<std::endl;
  if (useW) std::cout<<"Using weights "<<std::endl;
  if (shapeComp) std::cout<<"Shape comparison only "<<std::endl;
  /// 
  std::string hnameZj  = "hZjet_"+myvar;
  std::string hnameDB  = "hDB_"+myvar;
  std::string hnameSig = "hSig_"+myvar;
  std::string hnamettb = "httb_"+myvar;
  std::string hnamest  = "hst_"+myvar;
  std::string hnameWj  = "hWjet_"+myvar;
  std::string hnameData  = "hdata_"+myvar;
  std::string varname = myvar+">>"+hnameZj;
  //if (hout != 0) {delete hout; hout=0;}
  //TH1F* hout = new TH1F(hname.c_str(),hname.c_str(),200,1.,-1.);
  TH1F* hout = (TH1F*)gROOT->Get(hnameZj.c_str());
  if (hout != NULL) {
    delete hout;
    delete (TH1F*)gROOT->Get((hnameDB).c_str());
    delete (TH1F*)gROOT->Get((hnameSig).c_str());
    delete (TH1F*)gROOT->Get((hnameWj).c_str());
    delete (TH1F*)gROOT->Get((hnamettb).c_str());
    delete (TH1F*)gROOT->Get((hnamest).c_str());
    delete (TH1F*)gROOT->Get((hnameData).c_str());
    hout = NULL;
  }
  hout = new TH1F(hnameZj.c_str(),hnameZj.c_str(),nBins,xmin,xmax);
  TCanvas* c = new TCanvas(myvar.c_str(),myvar.c_str(),800,600);
  if (logx) c->SetLogx();
  if (logy) c->SetLogy();
  double xzj = fzjet->Draw(varname.c_str(), myCut.c_str());
  double xzjw = hout->GetSumOfWeights();
  //return;
  std::cout<<"Zjets   events            = "<<xzj<<" "<<hout->GetSumOfWeights()<<std::endl;
  
  TH1F* houtdb  = (TH1F*)hout->Clone();
  houtdb->Scale(0.);
  houtdb->SetName((hnameDB).c_str());
  houtdb->SetNameTitle((hnameDB).c_str(), (hnameDB).c_str());
  double xdb = fdb->Project((hnameDB).c_str(), myvar.c_str(), myCut.c_str(),"SAME");
  double xdbw = houtdb->GetSumOfWeights();
  std::cout<<"DiBoson events            = "<<xdb<<" "<<houtdb->GetSumOfWeights()<<std::endl;;
  TH1F* houtttb  = (TH1F*)hout->Clone();
  houtttb->Scale(0.);
  houtttb->SetName((hnamettb).c_str());
  houtttb->SetNameTitle((hnamettb).c_str(),(hnamettb).c_str());
  double xttb = fttbar->Project((hnamettb).c_str(), myvar.c_str(), myCut.c_str(),"SAME");
  double xttbw = houtttb->GetSumOfWeights();
  std::cout<<"ttbar events              = "<<xttb<<" "<<houtttb->GetSumOfWeights()<<std::endl;;
  TH1F* houtst  = (TH1F*)hout->Clone();
  houtst->Scale(0.);
  houtst->SetName((hnamest).c_str());
  houtst->SetNameTitle((hnamest).c_str(),(hnamest).c_str());
  double xst = fstop->Project((hnamest).c_str(), myvar.c_str(), myCut.c_str(),"SAME");
  double xstw = houtst->GetSumOfWeights();
  std::cout<<"single t events           = "<<xst<<" "<<houtst->GetSumOfWeights()<<std::endl;;
  TH1F* houtwj  = (TH1F*)hout->Clone();
  houtwj->Scale(0.);
  houtwj->SetName((hnameWj).c_str());
  houtwj->SetNameTitle((hnameWj).c_str(), (hnameWj).c_str());
  double xwj = fwjet->Project((hnameWj).c_str(), myvar.c_str(), myCut.c_str(),"SAME");
  double xwjw = houtwj->GetSumOfWeights();
  std::cout<<"Wjets events              = "<<xwj<<" "<<houtwj->GetSumOfWeights()<<std::endl;;  

  double totBkg = xst+xttb+xdb+xwj+xzj;
  double totBkgw = xstw+xttbw+xdbw+xwjw+xzjw;
  //double totBkgw = houtst->GetSumOfWeights()+houtttb->GetSumOfWeights()+houtdb->GetSumOfWeights()+houtwj->GetSumOfWeights()+hout->GetSumOfWeights();
  std::cout<<"Total background events   = "<<totBkg<<" "<<totBkgw<<std::endl;;
  TH1F* houtData  = (TH1F*)hout->Clone();
  houtData->Scale(0.);
  houtData->SetName((hnameData).c_str());
  houtData->SetNameTitle((hnameData).c_str(),(hnameData).c_str());
  double xData = fdata->Project((hnameData).c_str(), myvar.c_str(), myCut.c_str(),"SAME");
  std::cout<<"DATA events               = "<<xData<<" "<<houtData->GetSumOfWeights()<<std::endl;;

  TH1F* houtSig = (TH1F*)hout->Clone();
  houtSig->Scale(0.);
  houtSig->SetName((hnameSig).c_str());
  houtSig->SetNameTitle((hnameSig).c_str(),(hnameSig).c_str());
  double xsig=0.;
  if (showSignal) xsig = fSignal->Project((hnameSig).c_str(), myvar.c_str(), myCutS.c_str(),"SAME");
  std::cout<<"Signal events             = "<<xsig<<" "<<houtSig->GetSumOfWeights()<<std::endl;;
  //houtSig->Draw();
  //return;

  if (shapeComp){
    if (useW) {
      hout->Scale(1./hout->GetSumOfWeights());
      houtSig->Scale(1./houtSig->GetSumOfWeights());
      houtdb->Scale(1./houtdb->GetSumOfWeights());
      houtttb->Scale(1./houtdb->GetSumOfWeights());
      houtst->Scale(1./houtdb->GetSumOfWeights());
      houtwj->Scale(1./houtwj->GetSumOfWeights());
      houtData->Scale(1./houtData->GetEntries());
    }
    else{ 
      hout->Scale(1./hout->GetEntries());
      houtSig->Scale(1./houtSig->GetEntries());
      houtdb->Scale(1./houtdb->GetEntries());
      houtttb->Scale(1./houtttb->GetEntries());
      houtst->Scale(1./houtst->GetEntries());
      houtwj->Scale(1./houtwj->GetEntries());
      houtData->Scale(1./houtData->GetEntries());
    }
  }

  hout->SetLineColor(kBlack);
  houtSig->SetLineColor(kRed);
  houtdb->SetLineColor(kBlue);
  houtttb->SetLineColor(kMagenta);
  houtst->SetLineColor(8);
  houtwj->SetLineColor(kOrange);
  double maxy = hout->GetMaximum();


  if (shapeComp) {
    if (maxy < houtdb->GetMaximum())
      maxy = houtdb->GetMaximum();
    if (maxy < houtSig->GetMaximum())
      maxy = houtSig->GetMaximum();
  }
  else
    {
      maxy += houtdb->GetMaximum();
      maxy += houtSig->GetMaximum();
    }
  if (shapeComp) {
    hout->SetMaximum(maxy*1.1);
    hout->Draw("hist");
    houtdb->Draw("samehist");
    houtSig->Draw("samehist");
  }
  else
    {
      //houtdb->SetMaximum(maxy*1.1);
      //      houtdb->Draw("HIST");
      houtst-> Add(houtwj);
      houtttb->Add(houtst);
      houtdb->Add(houtttb);
      hout->Add(houtdb);
      //      hout->Draw("SAMEHIST");
      houtSig->Add(hout);
      if (houtSig->GetMaximum() < houtData->GetMaximum()) houtSig->SetMaximum((houtData->GetMaximum())*1.2) ;
      if (logy) houtSig->SetMaximum(2.*houtSig->GetMaximum());
      houtSig->SetTitle((tagSel+" "+scoreSignal+"  "+myvar).c_str());
      houtSig->Draw("HIST");
      hout->Draw("HISTSAME");
      houtdb->Draw("HISTSAME");
      houtttb->Draw("HISTSAME");
      houtst->Draw("HISTSAME");
      houtwj->Draw("HISTSAME");
      
      houtData->SetMarkerStyle(20);
      houtData->SetMarkerSize(0.75);
      if (showData) houtData->Draw("SAMEP");
    }

  TLegend * leg = new TLegend(0.55,0.6,0.9,0.85);

  char buffer[50];
  sprintf(buffer, "%10.1f Z+jets", xzjw);  
  leg->AddEntry(hout,    buffer,  "l");
  sprintf(buffer, "%10.1f VV", xdbw);  
  leg->AddEntry(houtdb,  buffer,      "l");
  sprintf(buffer, "%10.1f ttbar", xttbw);  
  leg->AddEntry(houtttb, buffer,   "l");
  sprintf(buffer, "%10.1f single t", xstw);  
  leg->AddEntry(houtst,  buffer,"l");
  sprintf(buffer, "%10.1f W+jets", xwjw);  
  leg->AddEntry(houtwj,  buffer,  "l");
  sprintf(buffer, "%10.1f Total SM", totBkgw);  
  leg->AddEntry(hout,    buffer,  "l");
  sprintf(buffer, "%10.0f Data", xData);  
  if (showData) leg->AddEntry(houtData,buffer,    "p");
  leg->SetTextAlign(12);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetFillColor(0);
  leg->SetTextFont(43);
  leg->SetTextSize(15);
  leg->Draw();


  std::ofstream outfile;
  //std::string fn = "/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/tmp/yield_"+tag+"_"+myvar+".txt";
  //std::string fn = "/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/tmp/yield_Run2_AllowOverlap_"+myvar+".txt";
  //std::string fn = "/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/tmp/yield_Run2_AllowOverlap_ZZandWZ.txt";
  //std::string fn = "/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/tmp/prova/yield_ScoreSignal"+tag+"_"+myvar+".txt";
  std::string fn = "yield_ScoreSignal"+tag+"_"+myvar+".txt";
  outfile.open(fn.c_str(), std::ios_base::app); // append instead of overwrite
  if (outfile.fail())
    {
      std::cout<<"Output file <"<<fn<<"> cannot be open"<<std::endl;
      throw std::ios_base::failure(std::strerror(errno));
    }
    //make sure write fails with exception if something is wrong
  outfile.exceptions(outfile.exceptions() | std::ios::failbit | std::ifstream::badbit);

  outfile<<"\nCut = "<<myCut<<std::endl;
  outfile<<"Zjets               = "<<xzj<<" "<<hout->GetSumOfWeights()<<std::endl;
  outfile<<"DiBoson             = "<<xdb<<" "<<houtdb->GetSumOfWeights()<<std::endl;;
  outfile<<"Wjets               = "<<xwj<<" "<<houtwj->GetSumOfWeights()<<std::endl;;  
  outfile<<"ttbar               = "<<xttb<<" "<<houtttb->GetSumOfWeights()<<std::endl;;
  outfile<<"single t            = "<<xst<<" "<<houtst->GetSumOfWeights()<<std::endl;;
  outfile<<"Total background    = "<<totBkg<<" "<<totBkgw<<std::endl;;
  outfile<<"DATA                = "<<xData<<" "<<houtData->GetSumOfWeights()<<std::endl;;
  outfile<<"Signal events       = "<<xsig<<" "<<houtSig->GetSumOfWeights()<<"      "<<scoreSignal<<std::endl;;

  outfile.close();

  
  if (savePlots) {
    //c->SaveAs(("~/html/allow_listing/DBL/tmp/prova/"+tagSel+"_ScoresFor"+scoreSignal+"_"+myvar+".pdf").c_str());
    //c->SaveAs(("~/html/allow_listing/DBL/tmp/prova/png/"+tagSel+"_ScoresFor"+scoreSignal+"_"+myvar+".png").c_str());
    c->SaveAs((tagSel+"_ScoresFor"+scoreSignal+"_"+myvar+".pdf").c_str());
    c->SaveAs((tagSel+"_ScoresFor"+scoreSignal+"_"+myvar+".png").c_str());
  }
  return 0;
  /*
  TCanvas* c1 = new TCanvas("c1","c1",800,600);
  TH2F* hbox = new TH2F("hbox","hbox",200,-5.,5.,200,0.0,.021);
  hbox->Draw();
  houtSig->Draw("histsame");
  hout->Draw("samehist");
  houtdb->Draw("samehist");
  */
  
  
}
