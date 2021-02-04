(* Wolfram Language package *)

BeginPackage["adpdOptics`",
	{
		"OpticaEM`",
		"adpdOptics`Components`",
		"adpdOptics`Simulators`"
	}]

RotMat3D::usage = "Generates a 3D rotation matrix about the X,Y,and Z axis given in radians"
RotateX::usage = "Generates a 3D rotation matrix by angle (in radians) about the X-axis"
RotateY::usage = "Generates a 3D rotation matrix by angle (in radians) about the Y-axis"
RotateZ::usage = "Generates a 3D rotation matrix by angle (in radians) about the Z-axis"

RotateAndMove::usage = "Rotates an Optica Component by the provided rotation matrix, then moves"

Begin["`Private`"]

RotMat3D[thetaX_,thetaY_,thetaZ_]:=
		{{1,0,0},{0,Cos[thetaX],-Sin[thetaX]},{0,Sin[thetaX],Cos[thetaX]}}.
		{{Cos[thetaY],0,Sin[thetaY]},{0,1,0},{-Sin[thetaY],0,Cos[thetaY]}}.
		{{Cos[thetaZ],-Sin[thetaZ],0},{Sin[thetaZ],Cos[thetaZ],0},{0,0,1}}

RotateX[theta_]:=RotMat3D[theta,0,0];
RotateY[theta_]:=RotMat3D[0,theta,0];
RotateZ[theta_]:=RotMat3D[0,0,theta];
		
RotateAndMove[components_, RotationMatrix_, Movement_] := 
  Move[Move[components, {0, 0, 0}, RotationMatrix], Movement];

End[]

EndPackage[]