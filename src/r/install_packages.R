# The rocker/r-ver base image pins CRAN to a dated snapshot,
# so these installs resolve to fixed package versions.
packages <- c("readr", "dplyr")
install.packages(packages, repos = getOption("repos"))

missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) stop("Failed to install: ", paste(missing, collapse = ", "))
cat("Installed packages:\n")
for (p in packages) cat(sprintf("  %s %s\n", p, as.character(packageVersion(p))))
