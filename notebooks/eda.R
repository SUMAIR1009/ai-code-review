# ============================================================
# CodeXGLUE Defect Detection (Devign) — EDA
# COM748 MSc Project — Muhammad Sumair (20098428)
# ============================================================

library(jsonlite)
library(ggplot2)

# ---- Load the three splits ----
load_jsonl <- function(path) {
  stream_in(file(path), verbose = FALSE)
}

train <- load_jsonl("train.jsonl")
valid <- load_jsonl("valid.jsonl")
test  <- load_jsonl("test.jsonl")

train$split <- "train"
valid$split <- "valid"
test$split  <- "test"

full <- rbind(train, valid, test)

# ---- Code length (characters and lines) ----
full$char_len <- nchar(full$func)
full$line_len <- lengths(gregexpr("\n", full$func)) + 1

# ---- Build the summary text ----
lines <- c()
add <- function(x) lines <<- c(lines, x)

add(strrep("=", 60))
add("CodeXGLUE Defect Detection (Devign) - EDA Summary")
add("COM748 MSc Project - Muhammad Sumair (20098428)")
add(strrep("=", 60))
add("")
add("--- SAMPLE COUNTS ---")
add(sprintf("  Train:      %s", format(nrow(train), big.mark = ",")))
add(sprintf("  Validation: %s", format(nrow(valid), big.mark = ",")))
add(sprintf("  Test:       %s", format(nrow(test),  big.mark = ",")))
add(sprintf("  TOTAL:      %s", format(nrow(full),  big.mark = ",")))
add("")

# Class distribution helper
dist <- function(df, name) {
  total <- nrow(df)
  clean <- sum(df$target == 0)
  buggy <- sum(df$target == 1)
  add(sprintf("--- CLASS DISTRIBUTION (%s) ---", name))
  add(sprintf("  Clean (target=0): %s  (%.1f%%)",
              format(clean, big.mark = ","), clean / total * 100))
  add(sprintf("  Buggy (target=1): %s  (%.1f%%)",
              format(buggy, big.mark = ","), buggy / total * 100))
  add(sprintf("  Imbalance ratio (clean:buggy): %.2f : 1", clean / buggy))
  add("")
}
dist(train, "TRAIN")
dist(full,  "OVERALL")

# Code length stats
add("--- CODE LENGTH (characters) ---")
add(sprintf("  Average:  %.0f", mean(full$char_len)))
add(sprintf("  Median:   %.0f", median(full$char_len)))
add(sprintf("  Shortest: %s", format(min(full$char_len), big.mark = ",")))
add(sprintf("  Longest:  %s", format(max(full$char_len), big.mark = ",")))
add("")
add("--- CODE LENGTH (lines) ---")
add(sprintf("  Average:  %.1f", mean(full$line_len)))
add(sprintf("  Median:   %.0f", median(full$line_len)))
add(sprintf("  Longest:  %s", format(max(full$line_len), big.mark = ",")))
add("")

# 512-token truncation proxy (CodeBERT limit)
over <- mean(full$char_len > 2000) * 100
add("--- TOKENISATION NOTE ---")
add(sprintf("  ~%.1f%% of functions exceed 2,000 chars (long functions", over))
add("  likely to hit CodeBERT's 512-token truncation limit).")
add("")

# Duplicate check (supports the methodology deduplication claim)
dups <- sum(duplicated(full$func))
add("--- DUPLICATE CHECK ---")
add(sprintf("  Exact-duplicate functions across full dataset: %s",
            format(dups, big.mark = ",")))
add("")

summary_text <- paste(lines, collapse = "\n")
cat(summary_text, "\n")

# ---- Save summary text ----
writeLines(summary_text, "eda_summary.txt")

# ---- Chart 1: class distribution bar chart ----
cls <- data.frame(
  Class = c("Clean (0)", "Buggy (1)"),
  Count = c(sum(train$target == 0), sum(train$target == 1))
)
p1 <- ggplot(cls, aes(x = Class, y = Count, fill = Class)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = format(Count, big.mark = ",")), vjust = -0.4, size = 4.2) +
  scale_fill_manual(values = c("Clean (0)" = "#1C7293", "Buggy (1)" = "#065A82")) +
  labs(title = "CodeXGLUE Train Split - Class Distribution",
       y = "Number of functions", x = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))
ggsave("class_distribution.png", p1, width = 7, height = 5, dpi = 150)

# ---- Chart 2: code length distribution ----
len_df <- data.frame(lines = pmin(full$line_len, 200))
p2 <- ggplot(len_df, aes(x = lines)) +
  geom_histogram(bins = 40, fill = "#1C7293", colour = "white") +
  labs(title = "Code Length Distribution (lines, capped at 200)",
       x = "Lines per function", y = "Frequency") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
ggsave("code_length_distribution.png", p2, width = 7, height = 5, dpi = 150)

cat("\nEDA complete. 3 files created:\n",
    " - eda_summary.txt\n",
    " - class_distribution.png\n",
    " - code_length_distribution.png\n")