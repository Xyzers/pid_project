@echo off
REM Script pour sauvegarder le travail quotidiennement avec Git ET Pousser sur GitHub

REM --- Configuration ---
REM Adapte le chemin vers ton projet si ce script n'est pas DANS ton dossier de projet
REM Exemple : cd /D C:\Jupyter\pid_project

echo ================================================================
echo  Sauvegarde Git & Push GitHub pour le projet pid_project
echo  (Lancez depuis C:\Jupyter\pid_project ou apres y avoir navigue)
echo ================================================================
echo.

REM Se placer dans le repertoire du projet si le batch est ailleurs
REM Si tu places ce fichier .bat directement dans C:\Jupyter\pid_project,
REM tu n'as pas besoin de la ligne 'cd' ci-dessous.
REM Sinon, decommente et adapte la ligne suivante :
REM cd /D C:\Jupyter\pid_project

REM Verifier si on est bien dans un repertoire Git
git rev-parse --is-inside-work-tree >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Ce dossier ne semble pas etre un depot Git, ou Git n'est pas accessible.
    echo Veuillez lancer ce script depuis votre invite de commandes git-cmd.exe
    echo ou directement depuis le dossier de votre projet Git.
    goto end
)

echo --- Statut Git Actuel ---
git status
echo.

echo --- Ajout de tous les fichiers modifies au commit ---
git add .
echo Fichiers ajoutes a la preparation.
echo.

REM Demander un message de commit a l'utilisateur
set "commit_message="
set /p commit_message="Entrez votre message de commit (ex: Avancement du jour): "

REM Verifier si le message est vide
if "%commit_message%"=="" (
    echo ATTENTION: Message de commit vide. Le commit et le push n'ont PAS ete effectues.
    goto end
)

echo.
echo --- Creation du commit local avec le message : "%commit_message%" ---
git commit -m "%commit_message%"

REM Verifier si le commit a reussi avant de tenter de pousser
if %errorlevel% neq 0 (
    echo ERREUR lors du commit local. Le push vers GitHub est annule.
    goto end
)
echo Commit local effectue avec succes.
echo.

echo --- Poussage des commits vers GitHub (origin main) ---
git push origin main

REM Verifier si le push a reussi
if %errorlevel% neq 0 (
    echo ATTENTION: Le PUSH vers GitHub a ECHOUE.
    echo Verifiez votre connexion internet ou les messages d'erreur ci-dessus.
    echo Vos modifications sont bien sauvegardees localement (commit effectue).
    goto end
)
echo Push vers GitHub (origin main) effectue avec succes.
echo.

echo ================================================================
echo Sauvegarde locale ET envoi vers GitHub termines avec succes !
echo ================================================================
echo.

:end
pause