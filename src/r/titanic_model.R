# Titanic survival prediction with logistic regression (R).
#
# Reads train.csv / test.csv from DATA_DIR (default: /app/data, mounted from src/data),
# prints every data-preparation step, fits a logistic regression (glm), reports accuracy,
# and writes predictions to DATA_DIR/predictions_r.csv.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

DATA_DIR <- Sys.getenv("DATA_DIR", "/app/data")
set.seed(42)

banner <- function(text) {
  cat("\n", strrep("=", 70), "\n", text, "\n", strrep("=", 70), "\n", sep = "")
}

load_csv <- function(name) {
  path <- file.path(DATA_DIR, name)
  if (!file.exists(path)) {
    cat(sprintf("[ERROR] %s not found.\n", path))
    cat("        Download the Kaggle Titanic data and place it in src/data/ (see README).\n")
    quit(status = 1)
  }
  df <- read_csv(path, show_col_types = FALSE)
  cat(sprintf("Loaded %s: %d rows x %d columns\n", name, nrow(df), ncol(df)))
  df
}

extract_title <- function(names) {
  t <- trimws(sub("^[^,]*,\\s*([^\\.]+)\\..*$", "\\1", names))
  t[t %in% c("Mlle", "Ms")] <- "Miss"
  t[t == "Mme"] <- "Mrs"
  t[!t %in% c("Mr", "Mrs", "Miss", "Master")] <- "Rare"
  t
}

print_counts <- function(x) {
  tb <- table(x)
  paste(sprintf("%s: %d", names(tb), as.integer(tb)), collapse = ", ")
}

prepare <- function(df, fill, label) {
  cat(sprintf("\n--- Preparing %s set ---\n", label))

  df$Title <- extract_title(df$Name)
  cat(sprintf("[ADD] Title extracted from Name -> %s\n", print_counts(df$Title)))

  n <- sum(is.na(df$Age))
  title_age <- fill$age_by_title[df$Title]
  title_age[is.na(title_age)] <- fill$age
  df$Age <- ifelse(is.na(df$Age), title_age, df$Age)
  cat(sprintf("[FILL] Age: %d missing values filled with the training median age per Title\n", n))

  n <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fill$fare
  cat(sprintf("[FILL] Fare: %d missing values filled with training median (%.2f)\n", n, fill$fare))

  n <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- fill$embarked
  cat(sprintf("[FILL] Embarked: %d missing values filled with training mode ('%s')\n", n, fill$embarked))

  df <- df %>% mutate(FamilySize = SibSp + Parch + 1)
  cat(sprintf("[ADD] FamilySize = SibSp + Parch + 1 (mean %.2f)\n", mean(df$FamilySize)))

  df <- df %>% mutate(IsAlone = as.integer(FamilySize == 1))
  cat(sprintf("[ADD] IsAlone = 1 if FamilySize == 1 (%d passengers travelling alone)\n", sum(df$IsAlone)))

  df <- df %>% mutate(Sex = as.integer(Sex == "female"))
  cat("[ENCODE] Sex: male -> 0, female -> 1\n")

  # Factors with fixed levels so train and test always get the same dummy columns
  df <- df %>% mutate(
    Pclass   = factor(Pclass, levels = c(1, 2, 3)),
    Embarked = factor(Embarked, levels = c("C", "Q", "S")),
    Title    = factor(Title, levels = c("Master", "Miss", "Mr", "Mrs", "Rare"))
  )
  cat("[ENCODE] Pclass, Embarked, Title converted to factors (glm creates dummy variables)\n")

  cat("[DROP] PassengerId, Name, Ticket, Cabin (identifiers / free text / >75% missing Cabin)\n")
  df %>% select(any_of("Survived"), Pclass, Sex, Age, SibSp, Parch, Fare,
                FamilySize, IsAlone, Embarked, Title)
}

accuracy <- function(actual, predicted) mean(actual == predicted)

# ---------------------------------------------------------------------------
banner("STEP 1: Load training data")
train <- load_csv("train.csv")
cat("\nFirst 5 rows:\n")
print(as.data.frame(head(train, 5)))

banner("STEP 2: Explore training data")
na_counts <- colSums(is.na(train))
cat("Missing values per column:\n")
print(na_counts[na_counts > 0])
cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
cat("\nSurvival rate by Sex:\n")
print(as.data.frame(train %>% group_by(Sex) %>% summarise(rate = round(mean(Survived), 3))))
cat("\nSurvival rate by Pclass:\n")
print(as.data.frame(train %>% group_by(Pclass) %>% summarise(rate = round(mean(Survived), 3))))

banner("STEP 3: Clean and engineer features (fill values learned from training data only)")
train_titles <- extract_title(train$Name)
fill <- list(
  age          = median(train$Age, na.rm = TRUE),
  age_by_title = tapply(train$Age, train_titles, median, na.rm = TRUE),
  fare         = median(train$Fare, na.rm = TRUE),
  embarked     = names(which.max(table(train$Embarked)))
)
train_df <- prepare(train, fill, "training")
cat(sprintf("\nFinal training columns: %s\n", paste(names(train_df), collapse = ", ")))

banner("STEP 4: Fit logistic regression")
# SibSp and Parch are left out because FamilySize = SibSp + Parch + 1 (perfectly collinear)
formula <- Survived ~ Pclass + Sex + Age + Fare + FamilySize + IsAlone + Embarked + Title

# Honest estimate: hold out 20% of train.csv (stratified by Survived)
val_idx <- unlist(lapply(split(seq_len(nrow(train_df)), train_df$Survived),
                         function(i) sample(i, round(0.2 * length(i)))))
fit_tr <- glm(formula, data = train_df[-val_idx, ], family = binomial)
val_pred <- as.integer(predict(fit_tr, train_df[val_idx, ], type = "response") > 0.5)
val_acc <- accuracy(train_df$Survived[val_idx], val_pred)
cat(sprintf("Hold-out validation accuracy (20%% of train.csv): %.4f\n", val_acc))

# Final model on all training data
model <- glm(formula, data = train_df, family = binomial)
cat("\nCoefficients (log-odds; negative = lower survival odds):\n")
print(round(coef(model), 3))

banner("STEP 5: Accuracy on the training set")
train_pred <- as.integer(predict(model, train_df, type = "response") > 0.5)
cat(sprintf("Training accuracy: %.4f\n", accuracy(train_df$Survived, train_pred)))

banner("STEP 6: Load test data and predict")
test <- load_csv("test.csv")
test_df <- prepare(test, fill, "test")
test_pred <- as.integer(predict(model, test_df, type = "response") > 0.5)
cat(sprintf("\nPredicted survivors in test set: %d of %d (%.3f)\n",
            sum(test_pred), length(test_pred), mean(test_pred)))

out_path <- file.path(DATA_DIR, "predictions_r.csv")
write_csv(data.frame(PassengerId = test$PassengerId, Survived = test_pred), out_path)
cat(sprintf("Predictions saved to %s\n", out_path))

banner("STEP 7: Accuracy on the test set")
if ("Survived" %in% names(test)) {
  cat(sprintf("Test accuracy: %.4f\n", accuracy(test$Survived, test_pred)))
} else {
  cat("Kaggle's test.csv has no 'Survived' column, so true test accuracy cannot be computed locally.\n")
  gs_path <- file.path(DATA_DIR, "gender_submission.csv")
  if (file.exists(gs_path)) {
    gs <- read_csv(gs_path, show_col_types = FALSE)
    merged <- inner_join(data.frame(PassengerId = test$PassengerId, Pred = test_pred), gs, by = "PassengerId")
    cat(sprintf("Agreement with Kaggle's gender_submission.csv baseline: %.4f\n",
                mean(merged$Pred == merged$Survived)))
    cat("(This is agreement with a baseline, NOT true accuracy.)\n")
  }
  cat(sprintf("Best local estimate of test accuracy = hold-out validation accuracy: %.4f\n", val_acc))
  cat("For the true score, upload predictions_r.csv to Kaggle.\n")
}

banner("DONE")
