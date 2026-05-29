"use client"

import * as React from "react"
import { useDropzone } from "react-dropzone"
import { UploadCloud, FileText, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface UploadZoneProps {
    onFileSelect: (file: File | null) => void
    selectedFile: File | null
}

export function UploadZone({ onFileSelect, selectedFile }: UploadZoneProps) {
    const onDrop = React.useCallback(
        (acceptedFiles: File[]) => {
            if (acceptedFiles.length > 0) {
                onFileSelect(acceptedFiles[0])
            }
        },
        [onFileSelect]
    )

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            "image/jpeg": [],
            "image/png": [],
        },
        maxFiles: 1,
        multiple: false,
    })

    // Preview URL
    const previewUrl = React.useMemo(() => {
        return selectedFile ? URL.createObjectURL(selectedFile) : null
    }, [selectedFile])

    React.useEffect(() => {
        // Cleanup URL on unmount or file change
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl)
        }
    }, [previewUrl])

    if (selectedFile) {
        return (
            <div className="relative w-full overflow-hidden rounded-xl border border-border bg-background shadow-sm">
                <div className="relative aspect-square w-full max-w-sm mx-auto overflow-hidden rounded-lg bg-secondary/20">
                    {previewUrl && (
                        <img
                            src={previewUrl}
                            alt="Preview"
                            className="h-full w-full object-contain p-4"
                        />
                    )}
                    <Button
                        variant="destructive"
                        size="icon"
                        className="absolute right-2 top-2 h-8 w-8 rounded-full shadow-md"
                        onClick={(e) => {
                            e.stopPropagation()
                            onFileSelect(null)
                        }}
                    >
                        <X className="h-4 w-4" />
                    </Button>
                </div>
                <div className="flex items-center justify-center p-2 text-sm text-muted-foreground">
                    <FileText className="mr-2 h-4 w-4" />
                    {selectedFile.name}
                </div>
            </div>
        )
    }

    return (
        <div
            {...getRootProps()}
            className={cn(
                "flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-muted-foreground/25 bg-muted/5 p-10 transition-all hover:border-primary/50 hover:bg-muted/30",
                isDragActive && "border-primary bg-primary/5"
            )}
        >
            <input {...getInputProps()} />
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-secondary shadow-sm">
                <UploadCloud className="h-10 w-10 text-muted-foreground" />
            </div>
            <h3 className="mt-4 text-lg font-semibold">Upload MRI Scan</h3>
            <p className="mt-2 text-center text-sm text-muted-foreground">
                Drag & drop or click to select a file <br />
                (JPEG or PNG, max 10MB)
            </p>
        </div>
    )
}
