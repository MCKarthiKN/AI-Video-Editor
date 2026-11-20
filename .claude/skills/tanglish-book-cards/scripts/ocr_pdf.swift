// OCR a scanned PDF with macOS's built-in Vision text recognition.
// Build once:  swiftc -O ocr_pdf.swift -o ocr_pdf
// Run:         ./ocr_pdf BOOK.pdf OUT.txt [firstPage] [lastPage]
// Output uses the same "[[P<n>]]" page markers as extract_pdf.py.
// About 1–2 seconds per page; run long books in the background.
import Foundation
import PDFKit
import Vision

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: ocr_pdf BOOK.pdf OUT.txt [first] [last]"); exit(2) }
let url = URL(fileURLWithPath: args[1])
let outPath = args[2]
let first = args.count > 3 ? Int(args[3])! : 1
let lastArg = args.count > 4 ? Int(args[4])! : Int.max
guard let doc = PDFDocument(url: url) else { print("cannot open PDF"); exit(1) }
let last = min(lastArg, doc.pageCount)
var out = ""
for i in first...last {
    autoreleasepool {
        guard let page = doc.page(at: i - 1) else { return }
        let box = page.bounds(for: .mediaBox)
        let scale: CGFloat = 2.5
        let w = Int(box.width * scale), h = Int(box.height * scale)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: ctx)
        guard let img = ctx.makeImage() else { return }
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = true
        try? VNImageRequestHandler(cgImage: img, options: [:]).perform([req])
        let lines = (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }
        out += "\n[[P\(i)]]\n" + lines.joined(separator: "\n") + "\n"
        if i % 20 == 0 { FileHandle.standardError.write("page \(i)\n".data(using: .utf8)!) }
    }
}
try! out.write(toFile: outPath, atomically: true, encoding: .utf8)
print("done", first, last, "of", doc.pageCount)
