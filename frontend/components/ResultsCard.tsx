import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AlertTriangle, CheckCircle, Download, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface PredictionResult {
    predicted_class: string
    confidence_pct: number
    class_probabilities: Record<string, number>
    warning?: string
}

export function ResultsCard({ result, file }: { result: PredictionResult, file: File }) {
    const [downloading, setDownloading] = React.useState(false)
    const [patientName, setPatientName] = React.useState("")
    const [patientAge, setPatientAge] = React.useState("")

    const handleDownloadReport = async () => {
        setDownloading(true)
        try {
            const formData = new FormData()
            formData.append("file", file)
            formData.append("predicted_class", result.predicted_class)
            formData.append("confidence_pct", result.confidence_pct.toString())
            if (patientName.trim()) formData.append("patient_name", patientName.trim())
            if (patientAge.trim()) formData.append("age", patientAge.trim())

            const res = await fetch("http://localhost:8000/generate-report", {
                method: "POST",
                body: formData,
            })

            if (!res.ok) {
                throw new Error("Failed to generate report")
            }

            const blob = await res.blob()
            const downloadUrl = window.URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = downloadUrl
            a.download = `Alzheimer_Medical_Report_${result.predicted_class}.pdf`
            document.body.appendChild(a)
            a.click()
            a.remove()
            window.URL.revokeObjectURL(downloadUrl)
        } catch (error) {
            console.error("Download failed:", error)
        } finally {
            setDownloading(false)
        }
    }

    const sortedProbs = Object.entries(result.class_probabilities)
        .sort(([, a], [, b]) => b - a)

    const isLowConfidence = result.confidence_pct < 60

    return (
        <Card className="h-full border-primary/20 bg-primary/5">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="text-white">Analysis Result</CardTitle>
                    <Badge variant={isLowConfidence ? "destructive" : "default"}>
                        {isLowConfidence ? "Low Confidence" : "High Confidence"}
                    </Badge>
                </div>
                <CardDescription className="text-white/70">
                    Hybrid Model Prediction (CNN + Swin Transformer)
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">

                {/* -- PRIMARY RESULT -- */}
                <div className="flex flex-col items-center justify-center space-y-2 rounded-lg bg-background p-6 shadow-sm border">
                    {isLowConfidence ? (
                        <AlertTriangle className="h-12 w-12 text-amber-500" />
                    ) : (
                        <CheckCircle className="h-12 w-12 text-primary" />
                    )}
                    <h2 className="text-2xl font-bold text-foreground">
                        {result.predicted_class}
                    </h2>
                    <div className="flex items-center gap-2 text-muted-foreground">
                        <span className="text-sm font-medium">Confidence:</span>
                        <span className="text-lg font-bold text-foreground">
                            {result.confidence_pct.toFixed(2)}%
                        </span>
                    </div>
                    {result.warning && (
                        <p className="mt-2 text-center text-xs text-amber-600 font-medium bg-amber-50 px-2 py-1 rounded">
                            {result.warning}
                        </p>
                    )}
                </div>

                {/* -- PROBABILITIES -- */}
                <div className="space-y-3">
                    <h4 className="text-sm font-medium text-white">Class Probabilities</h4>
                    {sortedProbs.map(([className, prob]) => (
                        <div key={className} className="space-y-1">
                            <div className="flex justify-between text-xs text-white/90">
                                <span className="font-medium">{className}</span>
                                <span>{prob.toFixed(1)}%</span>
                            </div>
                            <Progress value={prob} className="h-2" indicatorClassName={
                                className === result.predicted_class ? "bg-primary" : "bg-muted-foreground/30"
                            } />
                        </div>
                    ))}
                </div>

                {/* -- MANUAL INPUTS FOR REPORT -- */}
                <div className="space-y-3 mt-6 pt-4 border-t border-white/10">
                    <h4 className="text-sm font-medium text-white">Patient Details (for Report)</h4>
                    <div className="space-y-1">
                        <Label className="text-white/80 text-xs">Patient Name</Label>
                        <Input 
                            value={patientName}
                            onChange={(e) => setPatientName(e.target.value)}
                            placeholder="e.g. John Doe"
                            className="bg-black/20 text-white border-white/20 h-9"
                        />
                    </div>
                    <div className="space-y-1">
                        <Label className="text-white/80 text-xs">Patient Age</Label>
                        <Input 
                            type="number"
                            value={patientAge}
                            onChange={(e) => setPatientAge(e.target.value)}
                            placeholder="e.g. 70"
                            className="bg-black/20 text-white border-white/20 h-9"
                        />
                    </div>
                </div>

                {/* -- DOWNLOAD REPORT BUTTON -- */}
                <Button 
                    className="w-full mt-4" 
                    variant="default"
                    onClick={handleDownloadReport}
                    disabled={downloading}
                >
                    {downloading ? (
                        <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Generating PDF...
                        </>
                    ) : (
                        <>
                            <Download className="mr-2 h-4 w-4" />
                            Download Medical Report
                        </>
                    )}
                </Button>

            </CardContent>
        </Card>
    )
}
