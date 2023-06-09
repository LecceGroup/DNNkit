#!/bin/bash 

dryRun=false

dataType=$1
analysis=$2
channel=$3
signal=$4
mcType=$5
tag=`date +"%Y%m%d_%H_%M_%S"`
tag="${mcType}_${tag}"

## to compute time for the process 
startClock=`date +"%s"`


echo " "
echo "Starting at `date`"


if [ -z "$dataType" ] || [ -z "$analysis" ] || [ -z "$channel" ] || [ -z "$signal" ] || [ -z "$mcType" ]
#if [ -z "$dataType" ] 
then
    echo "Usage is: ./addScores.sh dataType analysis channel signal mcType"
    echo "dataType = data, Zjet, Wjet, ttbar, stop, diboson, all"
    echo "analysis = merged, resolved"
    echo "channel  = ggF or VBF"
    echo "signal   = Radion, RSG, HVTWZ"
    echo "mcType   = mc16a, mc16d, mc16e"
    exit
fi
echo tag = $tag
#dirScores=/scratch/stefania/${tag}
myscratcharea=`whoami`
dirScores=/scratch/${myscratcharea}/${tag}
if [ ! -d $dirScores ]
then
    mkdir -p $dirScores
fi
echo " "
echo "Running with ${dataType} ${analysis} ${channel} ${signal} ${mcType} ${tag}"

## input directory 
fileLocation="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_${mcType}_VV_2lep_PFlow_UFO/fetch/data-MVATree/"
## output directory
outFileLocation="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_${mcType}_VV_2lep_PFlow_UFO_Scores_loosePDNN/"
if [ ! -d "$outFileLocation" ]; then
    echo ERROR output directory ${outFileLocation} does not exist -  stop here
    exit
fi
if [ ! -d "$fileLocation" ]; then
    echo ERROR input directory ${fileLocation} does not exist -  stop here
    exit
fi
echo " "
echo input file Location = $fileLocation
echo output file Location = $outFileLocation
cd $fileLocation;
echo "in $PWD"
#ls -l $fileLocation
if [ $dataType == "all" ]; then
    if [ $channel == "ggF" ]; then 
	filelist=`ls data*.root Zjets*.root Wjet*.root ttbar*.root stop*.root Diboson*.root ${signal}*.root`
    elif [ $channel == "VBF" ]; then
	if [[ $signal == "VBF"* ]]; then
	    echo "ERROR Signal must be Radion, RSG, HVTWZ; prefix VBF will be added when channel is ${channel}"
	    exit
	fi
	signal="VBF${signal}"
	echo signal redefined as ${signal}
	#filelist=`ls data*.root Zjets*.root Wjet*.root ttbar*.root stop*.root Diboson*.root VBF${signal}*.root`
	# signal must be VBF*
	filelist=`ls data*.root Zjets*.root Wjet*.root ttbar*.root stop*.root Diboson*.root ${signal}*.root`
    fi
else
   filelist=`ls *.root | grep ${dataType}`
fi
echo "back to "
cd -
suffix=".root"
vbfstr="VBF"
echo Start loop over files ....
for inpFile in $filelist ; do
    echo inpFile=$inpFile 
    if [ $channel == "ggF" ]; then
	#echo "uffa"
	if [[ $inpFile == *"VBF"* ]]; then
	    echo ... skipping ${inpFile}
	    continue
	fi       
    elif [ $channel == "VBF" ]; then
	if [[ $inpFile == "Radion"* ]]; then # || [ $inpFile == "RSG"* ] || [ $inpFile == "HVT"* ] ]; then
	    echo ... skipping ${inpFile}
	    continue
	fi
	if [[ $inpFile == "RSG"* ]]; then 
	    echo ... skipping ${inpFile}
	    continue
	fi
	if [[ $inpFile == "HVT"* ]]; then 
	    echo ... skipping ${inpFile}
	    continue
	fi
    fi
    echo ... processing ${inpFile}

    myCmd="python computePNNScoresOnSelection.py -i $inpFile -a ${analysis} -c ${channel} -s ${signal} -m ${mcType} -t ${dirScores}"
    echo $myCmd
    if [ "$dryRun" = false ]; then
	$myCmd
    fi

    myCmd="root -b -q ./addPNNScores2treeOnSelection.C(\"${inpFile}\",\"${analysis}\",\"${channel}\",\"${signal}\",\"${mcType}\",\"${dirScores}\")"
    echo $myCmd
    if [ "$dryRun" = false ]; then
	$myCmd
    fi
    
    inpFile=${inpFile%"$suffix"}
    echo "redefined inpFile = ${inpFile}"
    
    myCmd="mv -f ${dirScores}/Scores_${inpFile}_${analysis}_${channel}_${signal}.root ${outFileLocation}/${analysis}/${channel}/${signal}/"
    echo $myCmd
    if [ "$dryRun" = false ]; then
	$myCmd
    fi
done

mynode=`cat /proc/sys/kernel/hostname`
mynodes=${mynode:0:5}
stopClock=`date +"%s"`
DeltaT="$(($stopClock-$startClock))"
strForLog="$0 $1 $2 $3 $4 $5 finished in ${DeltaT} s at `date` on node ${mynodes} tmp output in ${dirScores}"
echo $strForLog | cat >>/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/logs/addings-scores-33-22.log 
strForLog="Output in ${outFileLocation}/${analysis}/${channel}/${signal}"
echo $strForLog | cat >>/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/logs/addings-scores-33-22.log

echo "Clean up the temporary storage area ${dirScores} from text files with scores"
## cleanup scratch dir from text files of scores
rm -fv ${dirScores}/Scores*.txt

echo "All done at `date`"
echo "stop"


