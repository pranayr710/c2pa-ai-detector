import { type NextRequest, NextResponse } from "next/server"
import { writeFile, unlink } from "fs/promises"
import { join } from "path"
import { spawn } from "child_process"

/**
 * POST /api/verify
 * Handles image upload and verification workflow
 */
export async function POST(request: NextRequest) {
  const formData = await request.formData()
  const file = formData.get("file") as File

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 })
  }

  // Validate file is an image or video
  if (!file.type.startsWith("image/") && !file.type.startsWith("video/")) {
    return NextResponse.json({ error: "File must be an image or video" }, { status: 400 })
  }

  const tempDir = join(process.cwd(), "temp")
  const filename = `${Date.now()}-${file.name}`
  const filepath = join(tempDir, filename)

  try {
    // Save uploaded file
    const bytes = await file.arrayBuffer()
    await writeFile(filepath, Buffer.from(bytes))

    // Run Python verification script
    const result = await runVerification(filepath)

    return NextResponse.json(result, { status: 200 })
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Verification failed",
        has_seal: false,
        is_valid: false,
        source_type: "ERROR",
        confidence: 0.0,
        details: {},
      },
      { status: 500 },
    )
  } finally {
    // Cleanup temp file
    try {
      await unlink(filepath)
    } catch (e) {
      // Silently ignore cleanup errors
    }
  }
}

/**
 * Run Python seal verification script
 */
function runVerification(imagePath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const python = spawn("python3", ["scripts/verify_wrapper.py", imagePath])

    let output = ""
    let errorOutput = ""

    python.stdout.on("data", (data) => {
      output += data.toString()
    })

    python.stderr.on("data", (data) => {
      errorOutput += data.toString()
    })

    python.on("close", (code) => {
      try {
        const trimmedOutput = output.trim()

        if (!trimmedOutput) {
          reject(new Error("No output from verification script"))
          return
        }

        // Try to parse JSON, handling potential extra whitespace
        let result
        try {
          result = JSON.parse(trimmedOutput)
        } catch (parseError) {
          // If JSON parsing fails, try to extract JSON from output
          const jsonMatch = trimmedOutput.match(/\{[\s\S]*\}/)
          if (jsonMatch) {
            result = JSON.parse(jsonMatch[0])
          } else {
            reject(new Error(`Failed to parse verification result: ${trimmedOutput.substring(0, 100)}`))
            return
          }
        }

        if (code !== 0 && errorOutput) {
          // Even if exit code is non-zero, use the JSON result if available
          if (result && typeof result === "object") {
            resolve(result)
          } else {
            reject(new Error(errorOutput))
          }
        } else {
          resolve(result)
        }
      } catch (e) {
        reject(e)
      }
    })
  })
}
