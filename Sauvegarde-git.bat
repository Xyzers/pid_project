@echo off
REM Script pour sauvegarder le travail quotidiennement avec Git

REM --- Configuration ---
REM Adapte le chemin vers ton projet si ce script n'est pas DANS ton dossier de projet
REM Exemple : cd /D C:\Jupyter\pid_project

echo ================================================================
echo  Sauvegarde Quotidienne Git pour le projet
echo  (Assurez-vous que ce script est lance depuis le dossier
echo   C:\Jupyter\pid_project ou que le chemin y est configure)
echo ================================================================
echo.

REM Se placer dans le repertoire du projet si le batch est ailleurs
REM Si tu places ce fichier .bat directement dans C:\Jupyter\pid_project,
REM tu n'as pas besoin de la ligne 'cd' ci-dessous.
REM Sinon, decommente et adapte la ligne suivante :
REM cd /D C:\Jupyter\pid_project

REM Verifier si on est bien dans un repertoire Git (optionnel mais utile)
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
set /p commit_message="Entrez votre message de commit (ex: Avancement taches du jour): "

REM Verifier si le message est vide
if "%commit_message%"=="" (
    echo ATTENTION: Message de commit vide. Le commit n'a PAS ete effectue.
    goto end
)

echo.
echo --- Creation du commit avec le message : "%commit_message%" ---
git commit -m "%commit_message%"
echo.

echo ================================================================
echo Sauvegarde locale terminee avec succes !
echo.
echo Rappel : N'oubliez pas de faire 'git push origin main'
echo de temps en temps pour sauvegarder vos commits sur GitHub.
echo ================================================================
echo.

:end
pause