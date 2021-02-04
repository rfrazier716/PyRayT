(* ::Package:: *)

(* Wolfram Language Package *)

(*TODO: Write a Monte-Carlo model simulator leveraging turbotrace which also generates reports*)

BeginPackage["adpdOptics`Simulators`", { "OpticaEM`"}]
(* Exported symbols added here with SymbolName::usage *)  

OpticalCoupling::usage = "A parameter for SystemReports, generates a report of the coupling from the source onto each sensor"

CalculateOpticalLength::usage = "Measures the optical length for a traced system onto every detector in the system"
CalculateOpticalCoupling::usage = "Measures the optical coupling for a traced system onto every detector in the system"

PrepareADPDSystem::usage = "Prepares a turbotrace object that can be fed into other simulators, does not include sources"

Options[SystemReport] = {PreparedSystem -> OpticaEM`TurboTrace[{}], ReportType -> {OpticalCoupling}, DisplaySummary->True};
SystemReport::usage = "Performs a ray trace, iterating over every source in the system, and prints out the selected reports"

GenerateMonteCarloTrace::usage = "Creates a prepared turbotrace from an ADI Optical System Association, where the system components and detectors have been moved randomly according to DisplacementModel"

Options[MonteCarloReport] = {RegenerateSystem -> True, PreparedSystem -> TurboTrace[{}], ReportType -> OpticalLength, DisplacementArguments -> {0, 0, 0}, NumberOfRuns -> 1};
MonteCarloReport::usage = "Performs a system report, moving every component in the system according to DisplacementModel" 



Begin["`Private`"] (* Begin Private Context *) 



(* ::Subsubsection:: *)
(*Private Functions*)


GenerateSourceLabels[N_] := Table["S" <> ToString[j], {j, N}]

GenerateDetectorLabels[N_] := Table["D" <> ToString[j], {j, N}]

GenerateTableWithHeaders[data_, rowHeader_, columnHeader_] := 
 MapThread[
  Prepend, 
  {Prepend[data, columnHeader], Prepend[rowHeader, ""]}
  ]
  
GenerateOpticalCouplingReport[data_] := 
Module[{},
  Print["--Optical Coupling Summary--"];
  Print[
  	Grid[
  		GenerateTableWithHeaders[data,GenerateSourceLabels[Dimensions[data][[1]]], GenerateDetectorLabels[Dimensions[data][[2]]]],
  		Frame -> All
  	]
  ];
]

GenerateOpticalLengthReport[data_] := 
Module[{dataMuSigma},
	(*Make data a presentable form*)
	dataMuSigma = ArrayReshape[(ToString[#1] <> " \[PlusMinus] " <> ToString[#2] &) @@@ Flatten[data, 1], Dimensions[data][[;; 2]]]; 
	Print["--Optical Length Summary--"];
	Print[
		Grid[
			GenerateTableWithHeaders[dataMuSigma, GenerateSourceLabels[Dimensions[data][[1]]], GenerateDetectorLabels[Dimensions[data][[2]]]],
			Frame -> All
		]
	];
]

GenerateReportPrintout[data_, ReportTypes_] := 
Module[{ReportAssociation},
	ReportAssociation = Association[OpticalCoupling -> GenerateOpticalCouplingReport, OpticalLength -> GenerateOpticalLengthReport];
	If[Length[ReportTypes] != 0,
   		Table[ReportAssociation[ReportTypes[[j]]][data[[;; , j]]], {j, 1, Length[ReportTypes]}]
   		,
   		ReportAssociation[ReportTypes][data[[;; , 1]]]
	]
]



(* ::Subsubsection:: *)
(*Simulators*)


(*TODO: Add functions to read the following from a trace result:
- Percent Coupling
- Optical Path Length
- Incident Angle on Detector
- Heat map on detector
*)

(*
TODO: Write visualization functions
TODO: Write function to generate prepared TurboTrace system given an OpticalSystem Association as input
*)


PrepareADPDSystem[SimulationObject_] := OpticaEM`TurboTrace[SimulationObject[#] & /@ {"System", "Detectors", "Boundary"}];

CalculateCouplingEfficiency[TracedSystem_, SimulationObject_] :=
Module[{GeneratedPower},
	GeneratedPower = Total@ReadTurboRays[TracedSystem, Intensity, {IntersectionNumber -> 1}];
  	Table[
   		Total@ReadRays[TracedSystem, Intensity, ComponentNumber -> (Length[Flatten[SimulationObject["System"]]] + j)]/GeneratedPower,
   		{j, 1, Length[SimulationObject["Detectors"]]}
   	]
]


CalculateOpticalLength[TracedSystem_, SimulationObject_] := 
Table[
  	{If[Length[#] > 0, Mean[#], -1], If[Length[#] > 1, StandardDeviation[#], 0]} &[ReadRays[TracedSystem, OpticalLength, ComponentNumber -> (Length[Flatten[SimulationObject["System"]]] + j)]]
 ,{j, 1, Length[SimulationObject["Detectors"]]}]


SystemReport[SimulationObject_, regenerateTrace_: True, OptionsPattern[]] :=
Module[{preparedSystem, simulatedTrace, nSources, nDetectors, ReportAssociation, reportData},
	ReportAssociation = Association[OpticalCoupling -> CalculateCouplingEfficiency, OpticaEM`OpticalLength -> CalculateOpticalLength];
  	nSources = Length[SimulationObject["Sources"]];
  	nDetectors = Length[SimulationObject["Detectors"]];
  	
  	If[DisplaySummary,
  		Print["Found " <> ToString[nSources] <> " Source(s) and " <> ToString[nDetectors] <> " Detector(s)"];
  	];
  
  	(*Do a Turboplot of the system,if provided do not regenerate*)
  	preparedSystem = If[regenerateTrace, 
    	OpticaEM`TurboTrace[SimulationObject[#] & /@ {"System", "Detectors", "Boundary"}]
    	, 
    	OptionValue[PreparedSystem]
    ];
    
    (*Iterate over sources and perform a turbo Trace, apply report functions on every trace*)
 	reportData = Table[
    	simulatedTrace = OpticaEM`TurboTrace[source, preparedSystem, OpticaEM`OutputType -> OpticaEM`TurboSystem];
    	(*Apply Reports via the ReportAssociation object (like a function pointer/dict)*)
    	Table[
     		ReportAssociation[key][simulatedTrace, SimulationObject],
     		{key, If[Length[#] == 0, {#}, #] &[OptionValue[ReportType]]}
 		],
 		{source, SimulationObject["Sources"]}
 	];
  	(*Generate the report printout*)
  	If[OptionValue[DisplaySummary],
  		GenerateReportPrintout[reportData, OptionValue[ReportType]]
  	];
  	
  	(*Return the coupling value and the generated system*)
  	{reportData, preparedSystem} 
]

(*Monte-Carlo Trace creates a prepared system with components moved according to displacement model*)
Options[GenerateMonteCarloTrace]= {DisplacementArguments->{0,0,0}};
GenerateMonteCarloTrace[SimulationModel_, DisplacementModel_ , opts : OptionsPattern[]] :=
	TurboTrace[
		{	
			Move[#, DisplacementModel @@ OptionValue[DisplacementArguments]] & /@ Join[(SimulationModel[#] & /@ {"System", "Detectors"})],
			SimulationModel["Boundary"]
		}, 
		Output -> TurboSystem
	]


MonteCarloReport[SimulationObject_, DisplacementModel_, opts : OptionsPattern[]] :=
Module[{tempObject, preppedSys,results},
	tempObject = SimulationObject;
	results=Table[
	  	If[OptionValue[RegenerateSystem],
	   		preppedSys = adpdOptics`Simulators`GenerateMonteCarloTrace[tempObject, DisplacementModel, DisplacementArguments -> OptionValue[DisplacementArguments]],
	   		preppedSys = OptionValue[PreparedSystem]
	   	];
	  	(*Randomly distribute sources*)
	  	AssociateTo[tempObject, "Sources" -> (Move[#, DisplacementModel @@ OptionValue[DisplacementArguments]] & /@ SimulationObject["Sources"])];
	  	
	  	(*Run Ray Trace with updated simulation object and return results, as well as sources in shuffled position*)
	  	SystemReport[ tempObject, False, PreparedSystem -> preppedSys, ReportType -> OptionValue[ReportType], DisplaySummary->False][[1]]
  	,OptionValue[NumberOfRuns]];
  	If[OptionValue[NumberOfRuns]==1,results[[1]],results]
] 


(*Don't use this function anymore, use SystemReport*)
Options[TraceOpticalCoupling] = {PreparedSystem -> TurboTrace[{}]};
TraceOpticalCoupling[SimulationObject_, regenerateTrace_: True, OptionsPattern[]] :=
 Module[{preparedSystem, simulatedTrace, totalRadiatedIntensity, nSources, nDetectors, coupling, DetectorRays},
  If[Length[SimulationObject["Sources"]] == 0, 
   SimulationObject["Sources"] = {SimulationObject["Sources"]}];
  If[Length[SimulationObject["Detectors"]] == 0, 
   SimulationObject["Detectors"] = {SimulationObject["Detectors"]}];
  nSources = Length[SimulationObject["Sources"]];
  nDetectors = Length[SimulationObject["Detectors"]];
  Print["Found " <> ToString[nSources] <> " Source(s) and " <> 
    ToString[nDetectors] <> " Detector(s)"];
  (*Do a Turboplot of the system, if provided do not regenerate*)
  
  preparedSystem = If[regenerateTrace,
    TurboTrace[
     SimulationObject[#] & /@ {"System", "Detectors", "Boundary"}],
    OptionValue[PreparedSystem]
    ];
  coupling = Table[
    simulatedTrace = TurboTrace[source, preparedSystem];
    totalRadiatedIntensity = Total@ReadRays[source, Intensity];
    DetectorRays = Table[
      Total@
       ReadRays[simulatedTrace, Intensity, 
        ComponentNumber -> (Length[SimulationObject["System"]] + j)],
      {j, 1, Length[SimulationObject["Detectors"]]}];
    DetectorRays/totalRadiatedIntensity
    , {source, SimulationObject["Sources"]}];
  Print["Trace Results:"];
  Print[Grid[
    GenerateTableWithHeaders[coupling, GenerateSourceLabels[nSources],
      GenerateDetectorLabels[nDetectors]], Frame -> All]];
  {coupling, 
   preparedSystem} (*Return the coupling value and the generated \
system*)
  ]

End[] (* End Private Context *)

EndPackage[]
