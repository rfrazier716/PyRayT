BeginTestSection["TestSources"]

BeginTestSection["LED Tests"]

BeginTestSection["The Number of Rays Generated should be equal to the \[OpenCurlyDoubleQuote]NumberOfRays\[CloseCurlyDoubleQuote] Option"]

VerificationTest[(* 1 *)
	CompoundExpression[Set[led, adpdOptics`components`sources`LED[1, 1, Rule[OpticaEM`NumberOfRays, 20]]], Length[OpticaEM`ReadRays[led, OpticaEM`RayLabels]]]
	,
	20	
	,
	TestID->"LEDGeneratedNumberOfElements"
]

EndTestSection[]

BeginTestSection["For a zero dimension input aperture, the rays should be distributed in a circle on the YZ plane with radius aperture/2"]

VerificationTest[(* 2 *)
	CompoundExpression[Set[apertureSize, 3], Set[led, adpdOptics`components`sources`LED[apertureSize, 1, Rule[OpticaEM`NumberOfRays, 1000]]], Set[rayOrigins, Part[OpticaEM`ReadRays[led, OpticaEM`RayStart], Span[1, All], List[2, 3]]], Apply[And, Map[Function[LessEqual[Norm[Slot[1]], Times[apertureSize, Power[2, -1]]]], rayOrigins]]]
	,
	True	
	,
	TestID->"Rays Generated Within Circular Aperture"
]

EndTestSection[]

BeginTestSection["For a 2 dimensional aperture, the rays should be distributed in a rectangle with sideLengths {x,y}"]

VerificationTest[(* 3 *)
	CompoundExpression[Set[aperture, List[3, 10]], Set[led, adpdOptics`components`sources`LED[aperture, 1, Rule[OpticaEM`NumberOfRays, 1000]]], Set[rayOrigins, Part[OpticaEM`ReadRays[led, OpticaEM`RayStart], Span[1, All], List[2, 3]]], Set[xBoundsFollowed, Apply[And, Map[Function[Less[Slot[1], Times[Part[aperture, 1], Power[2, -1]]]], Abs[Part[rayOrigins, Span[1, All], 1]]]]], Set[yBoundsFollowed, Apply[And, Map[Function[Less[Slot[1], Times[Part[aperture, 2], Power[2, -1]]]], Abs[Part[rayOrigins, Span[1, All], 2]]]]], List[xBoundsFollowed, yBoundsFollowed]]
	,
	List[True, True]	
	,
	TestID->"9f012dce-979c-4102-909c-cd6cac401714"
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
