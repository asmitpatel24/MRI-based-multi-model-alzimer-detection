"use client"

import * as React from "react"
import { UploadZone } from "@/components/UploadZone"
import { ResultsCard } from "@/components/ResultsCard"
import { Button } from "@/components/ui/button"
import { Loader2, BrainCircuit, Activity } from "lucide-react"

export default function Dashboard() {
  const [file, setFile] = React.useState<File | null>(null)

  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<any | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const handlePredict = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const res = await fetch(`http://localhost:8000/predict`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Error: ${res.statusText}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err: any) {
      console.error(err)
      setError(err?.message || "Failed to predict")
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <main className="min-h-screen bg-slate-50 dark:bg-slate-950 p-4 md:p-8">
      <div className="mx-auto max-w-6xl space-y-8">

        {/* -- HEADER -- */}
        <header className="flex items-center justify-between pb-6 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="bg-primary/10 p-2 rounded-lg">
              <BrainCircuit className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                Alzheimer Detection
              </h1>
              <p className="text-sm text-white/80">
                (CNN + Swin Transformer)
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-white bg-slate-900 px-3 py-1 rounded-full border border-slate-800 shadow-sm">
            <Activity className="h-4 w-4 text-green-500" />
            <span>Model Active</span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* -- LEFT COLUMN: UPLOAD & INPUTS -- */}
          <div className="lg:col-span-2 space-y-6">
            <section className="space-y-4">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-white">
                1. Upload MRI Scan
              </h2>
              <UploadZone
                selectedFile={file}
                onFileSelect={(f) => {
                  setFile(f)
                  setResult(null) // reset result on new file
                }}
              />
            </section>

            <div className="pt-4">
              <Button
                size="lg"
                className="w-full text-lg h-12"
                onClick={handlePredict}
                disabled={!file || loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Processing Hybrid Model...
                  </>
                ) : (
                  "Run Analysis"
                )}
              </Button>
              {error && (
                <p className="mt-2 text-sm text-center text-destructive font-medium">
                  {error}
                </p>
              )}
            </div>
          </div>

          {/* -- RIGHT COLUMN: RESULTS -- */}
          <div className="lg:col-span-1">
            <h2 className="text-lg font-semibold mb-4 text-white">2. Results</h2>
            {result ? (
              <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                <ResultsCard result={result} file={file!} />
                <Button
                  variant="outline"
                  className="w-full mt-4"
                  onClick={handleReset}
                >
                  Start New Analysis
                </Button>
              </div>
            ) : (
              <div className="h-full min-h-[400px] rounded-xl border-2 border-dashed border-muted bg-muted/20 flex flex-col items-center justify-center text-muted-foreground p-8 text-center">
                <BrainCircuit className="h-12 w-12 mb-4 opacity-20" />
                <p>Prediction results will appear here after analysis.</p>
              </div>
            )}
          </div>

        </div>
      </div>
    </main>
  )
}
