(* ::Package:: *)

(*TODO: Rename this components and put in the main directory*)

(*Additional components to make

> SiliconWindow -- pretty much just a window specify refractive index

> Redefine the silicon NK data to extend to 5um -- Using Fab Values (SiliconADI)
> Add a Germanium Model to 4.5 (GermainumADI)
> Maybe glass if needed 

> Detector? (wrap around Baffle)
> Check for CAD import/export
*)

(* Wolfram Language package *)
BeginPackage["adpdOptics`Components`",{"OpticaEM`"}]

(*Protect Option Parameters*)
RayPrefix
ThetaMax

(*Quick Descriptors*)
RayPrefix::usage="Used for Source creation to put a prefix before each Ray's label"
ThetaMax::usage="The maximum cone angle to generate rays for a lambertian source"

OffAxisParabolicMirror::usage = "Created an off axis parabaloid with a given focus length, aperture is centered around the off-axis focal point. Focal length is at the origin"
ADIEllipticalMirror::usage = "An elliptical mirrored surface, with major an minor axes set by the axes argument, aperture is a projection from spherical coordinates"
XYEllipticalMirror::usage = "An elliptical mirrored surface, with major an minor axes set by the axes argument, aperture is a projection from spherical coordinates"

LED::usage = "Creates a Lambertian Distributed LED Model"
PseudoLED::usage = "Creates a Lambertian LED model where the angles are uniformaly distributed about ThetaMax, Ray origins are randomly placed about LED surface"

GratingWithPinHole::usage = "Creates a grating with an aperture on the opposite side of the substrate"


(* ::Subsection:: *)
(*Implementation*)


Begin["`Private`"]
(* Implementation of the package *)


(* ::Subsubsection:: *)
(*Dependencies*)


(* Required Random Sampling Functions for Generating LED*)
SphereSample[n_: 1, r_:1, thetaMax_: \[Pi]] := If[n==1,#[[1]],#]&[
 Map[FromSphericalCoordinates[{r, 
     ArcCos[1 - #[[1]] (1 - Cos[thetaMax])], 
     2 \[Pi] (#[[2]] - 0.5)}] &, RandomReal[{0, 1}, {n, 2}]]
	]
	
RectangleSample[n_: 1, dimensions_: {1, 1}] := 
  If[n == 1, #[[1]], #] &[
   Map[{dimensions[[1]]*#[[1]], dimensions[[2]]*#[[2]]} &, 
    RandomReal[{0, 1}, {n, 2}] - 0.5]];
    
DiskSample[n_: 1, radius_: 1] := 
 If[n == 1, #[[1]], #] &[
  Map[FromPolarCoordinates[{radius Sqrt[#[[1]]], 
      2 \[Pi] (#[[2]] - 0.5)}] &, RandomReal[{0, 1}, {n, 2}]]]


SolidAngleSample[n_, angleMax_] := Module[{sphereCoverage, nSamples},
  sphereCoverage = 0.5 (1 - Cos[angleMax]);
  nSamples = 
   Round[n/sphereCoverage]; (*Scale samples because we throw out a \
lot*)
  Pick[#, 
     Thread[Function[{xProjection, theta}, 
        ArcCos[xProjection] <= theta][#[[;; , 1]], angleMax]]] &[
   SpherePoints[nSamples]]
  ]


(* ::Subsubsection:: *)
(*Mirrors*)


OffAxisParabolicMirror[
focallength_,
aperture_,
thickness_:0,
 opts___]:=
Module[{xm,lowerheight},
If[Length[aperture]<2,
xm=focallength-aperture/2;
lowerheight=1/(2 focallength)*xm^2;
,
xm=focallength-aperture[[1]]/2;
lowerheight=1/(2 focallength)*xm^2;
];
Move[ThinParabolicMirror[-0.5*focallength,aperture,-lowerheight+thickness,OffAxis->{0,focallength},opts]]
]

ADIEllipticalMirror[axes_, aperture_, opts___] := 
 CustomMirror[
  Function[{\[Theta], \[Phi]}, {axes[[2]]*Cos[\[Theta]], 
    axes[[2]]*Sin[\[Theta]] Sin[\[Phi]], 
    axes[[1]]*Sin[\[Theta]] Cos[\[Phi]]}], aperture, "EllipticalMirror", 
  opts]
  
XYEllipticalMirror[axes_, aperture_, opts___] := 
 CustomMirror[
  Function[{s, 
    t}, {axes[[1]](1- Sqrt[
       1. - (t^2/axes[[2]]^2 + s^2/axes[[1]]^2)]), s, t}], aperture, 
  "XYEllipticalMirror", opts]



(* ::Subsubsection:: *)
(*Sources*)


Options[LED] = Join[Options[Ray], {NumberOfRays -> 100, ThetaMax -> \[Pi]/2., RayPrefix -> ""}];
LED[Aperture_, thickness_, opts : OptionsPattern[LED]] :=
 Module[{samplePositions, sampleIntensity, sampleAngles, ledGraphic, filteredRules},
  If[Length[Aperture] == 0,
   samplePositions = DiskSample[OptionValue[NumberOfRays], Aperture/2.];
   ledGraphic = BoxGraphic[{0, -Aperture/2., -Aperture/2.}, {thickness, Aperture/2., Aperture/2.}, "LED"];
   ,
   samplePositions = RectangleSample[OptionValue[NumberOfRays], Aperture];
   ledGraphic = BoxGraphic[{0, -0.5 Aperture[[1]], -0.5 Aperture[[2]]}, {thickness, 0.5 Aperture[[1]], 0.5 Aperture[[2]]}, "LED"];
   ];
  sampleAngles = {#[[3]], #[[1]], #[[2]]} & /@ SphereSample[OptionValue[NumberOfRays], 1, OptionValue[ThetaMax]];
  (*sampleIntensity = 100./\[Pi] ArcTan[Sqrt[#[[3]]^2 + #[[2]]^2]/#[[1]]] & /@ sampleAngles;*)
  sampleIntensity = 100./Sqrt[1 + (#1[[3]]^2 + #1[[2]]^2)/#1[[1]]^2] & /@ sampleAngles;
  filteredRules = FilterRules[{opts}, Options[Ray]];
  {
  	ledGraphic, 
  	CustomRays[
  		Join[{
      		{RayStart, Join[{thickness}, #] & /@ samplePositions},
      		{RayLabels, OptionValue[RayPrefix] <> "Ray" <> ToString[#] & /@ Range[OptionValue[NumberOfRays]]},
      		{RayTilt, sampleAngles}, {Intensity, sampleIntensity}
      		},
     		{#, Table[ # /. filteredRules, OptionValue[NumberOfRays]]} & /@ Keys[filteredRules]
 		], 
 	ResolvePoints -> True]
  }]
  
Options[PseudoLED] = Join[Options[Ray], {NumberOfRays -> 100, ThetaMax -> \[Pi]/2., RayPrefix -> ""}];
PseudoLED[Aperture_, thickness_, opts : OptionsPattern[LED]] :=
 Module[{samplePositions, sampleIntensity, sampleAngles, ledGraphic, filteredRules, nPickedRays},
	sampleAngles = SolidAngleSample[OptionValue[NumberOfRays], OptionValue[ThetaMax]];
	nPickedRays = Length[sampleAngles];
  If[Length[Aperture] == 0,
   samplePositions = DiskSample[nPickedRays, Aperture/2.];
   ledGraphic = BoxGraphic[{0, -Aperture/2., -Aperture/2.}, {thickness, Aperture/2., Aperture/2.}, "LED"];
   ,
   samplePositions = RectangleSample[nPickedRays, Aperture];
   ledGraphic = BoxGraphic[{0, -0.5 Aperture[[1]], -0.5 Aperture[[2]]}, {thickness, 0.5 Aperture[[1]], 0.5 Aperture[[2]]}, "LED"];
   ];
  
  (*sampleIntensity = 100./\[Pi] ArcTan[Sqrt[#[[3]]^2 + #[[2]]^2]/#[[1]]] & /@ sampleAngles;*)
  sampleIntensity = 100./Sqrt[1 + (#1[[3]]^2 + #1[[2]]^2)/#1[[1]]^2] & /@ sampleAngles;
  filteredRules = FilterRules[{opts}, Options[Ray]];
  {
  	ledGraphic, 
  	CustomRays[
  		Join[{
      		{RayStart, Join[{thickness}, #] & /@ samplePositions},
      		{RayLabels,{ OptionValue[RayPrefix] <> "Ray" <> ToString[#]} & /@ Range[nPickedRays]},
      		{RaySourceNumber, {{#,0},{0,0}}&/@Range[nPickedRays]},
      		{RayTilt, sampleAngles}, {Intensity, sampleIntensity}
      		},
     		{#, Table[ # /. filteredRules, nPickedRays]} & /@ Keys[filteredRules]
 		], 
 	ResolvePoints -> True]
  }]
  


(* ::Subsubsection:: *)
(*Miscellaneous*)


Options[GratingWithPinHole] = Join[Options[Grating], Options[PinHole]];
GratingWithPinHole[gratingFunction_, aperture_, pinholeAperture_, thickness_, opts : OptionsPattern[]] :=
	{
  		PinHole[aperture, pinholeAperture, FilterRules[{opts}, Options[PinHole]]],
  		adpdOptics`RotateAndMove[
  			Grating[gratingFunction, aperture, thickness, FilterRules[{opts}, Options[Grating]]]
  			,adpdOptics`RotMat3D[0, 180 Degree, 0], thickness
  		]
  	}


       
End[]
EndPackage[]
