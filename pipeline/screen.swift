// Screens frames from inside each video (YouTube's hq1/hq2/hq3, a quarter, half and three quarters in) with
// Apple's Vision framework, so the opening only plays clips of real people: faces present, no drawings or
// animation, and no frames covered in on-screen text. Runs locally; nothing leaves the laptop.
// swift screen.swift <frames dir> <out.json>
import AppKit
import Vision

let args = CommandLine.arguments
let dir = args[1], out = args[2]
let DRAWN: Set<String> = ["illustrations", "drawing", "cartoon", "sketch", "painting", "comics", "animation", "anime", "art"]

func cg(_ path: String) -> CGImage? {
    guard let img = NSImage(contentsOfFile: path) else { return nil }
    var r = CGRect(origin: .zero, size: img.size)
    return img.cgImage(forProposedRect: &r, context: nil, hints: nil)
}

var result: [String: [String: Any]] = [:]
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
for f in files {
    let id = String(f.dropLast(8)), path = dir + "/" + f   // "<id>_hq2.jpg"
    guard let image = cg(path), image.width > 120 else { continue }   // 120 px wide is YouTube's placeholder for a missing frame
    // the frames are 4:3 with the 16:9 picture letterboxed inside; only look at the picture
    let crop = image.cropping(to: CGRect(x: 0, y: image.height / 8, width: image.width, height: image.height * 3 / 4)) ?? image
    let faces = VNDetectFaceRectanglesRequest(), text = VNRecognizeTextRequest(), labels = VNClassifyImageRequest()
    text.recognitionLevel = .fast
    try? VNImageRequestHandler(cgImage: crop).perform([faces, text, labels])
    let faceArea = (faces.results ?? []).map { $0.boundingBox.width * $0.boundingBox.height }.max() ?? 0
    let textArea = (text.results ?? []).filter { $0.confidence > 0.4 }.reduce(0.0) { $0 + $1.boundingBox.width * $1.boundingBox.height }
    let drawn = (labels.results ?? []).filter { DRAWN.contains($0.identifier) }.map { Double($0.confidence) }.max() ?? 0
    var row = result[id] ?? [:]
    var frames = row["frames"] as? [[String: Double]] ?? []
    frames.append(["face": Double(faceArea), "text": Double(textArea), "drawn": drawn])
    row["frames"] = frames
    result[id] = row
}
let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
try data.write(to: URL(fileURLWithPath: out))
print("\(result.count) videos screened")
