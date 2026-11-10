import { AbsoluteFill, Composition, registerRoot, useCurrentFrame } from "raea";

// A transparent split wipe. Put the underlying edit between frames 8 and 9.
function Wipe() {
  const frame = useCurrentFrame();
  const travel = Math.min(frame, 17 - frame) / 8;
  const coverage = travel * travel * (3 - 2 * travel);
  const width = 480 * coverage;

  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: 0, top: 0, width, height: 540, backgroundColor: "#10343c" }} />
      <div style={{ position: "absolute", right: 0, top: 0, width, height: 540, backgroundColor: "#10343c" }} />
    </AbsoluteFill>
  );
}

// A standalone scene with an animated horizon and a chapter heading.
function Scene() {
  const frame = useCurrentFrame();
  const rise = 310 - Math.min(frame, 26) * 3;
  const opacity = Math.min(frame / 18, 1);

  return (
    <AbsoluteFill style={{ backgroundColor: "#102733", color: "#f2e8d3" }}>
      <div style={{ position: "absolute", top: rise, left: 0, width: 960, height: 540 - rise, backgroundColor: "#21616b" }} />
      <div style={{ position: "absolute", top: rise - 4, left: 0, width: 960, height: 4, backgroundColor: "#74d4bb" }} />
      <div style={{ position: "absolute", top: 336, left: 84, fontSize: 56, fontWeight: 700, opacity }}>NEXT STOP</div>
    </AbsoluteFill>
  );
}

registerRoot(() => (
  <>
    <Composition id="wipe" component={Wipe} width={960} height={540} fps={30} durationInFrames={18} />
    <Composition id="scene" component={Scene} width={960} height={540} fps={30} durationInFrames={60} />
  </>
));
