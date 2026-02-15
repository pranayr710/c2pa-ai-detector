import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"

/**
 * GET /api/registry
 * Returns the trusted sources registry
 */
export async function GET(request: NextRequest) {
  try {
    const registry = await loadRegistry()
    return NextResponse.json(registry, { status: 200 })
  } catch (error) {
    console.error("Registry error:", error)
    return NextResponse.json({ error: "Failed to load registry" }, { status: 500 })
  }
}

/**
 * Load registry from Python script
 */
function loadRegistry(): Promise<Record<string, any>> {
  return new Promise((resolve, reject) => {
    const python = spawn("python3", ["scripts/get_registry.py"])

    let output = ""
    let errorOutput = ""

    python.stdout.on("data", (data) => {
      output += data.toString()
    })

    python.stderr.on("data", (data) => {
      errorOutput += data.toString()
    })

    python.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(errorOutput || "Registry script failed"))
        return
      }

      try {
        const registry = JSON.parse(output)
        resolve(registry)
      } catch (e) {
        reject(new Error("Failed to parse registry"))
      }
    })
  })
}
