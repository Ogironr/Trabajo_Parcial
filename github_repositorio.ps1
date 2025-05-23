<#
Install-Module posh-git -Scope CurrentUser -Force
Import-Module posh-git
#>

#------------------------------------------------------------ GITHUB

function CrearRepositorioGithub {
    param (
        [string]$username,
        [string]$name,
        [string]$email,
        [string]$nombre_repo
    )

    try {
        git config --global user.username $username
        git config --global user.name $name
        git config --global user.email $email

        gh repo create $nombre_repo --public
        
        git init -b main
        git branch -M main
        $remote = "https://github.com/Ogironr/" + $nombre_repo
        git remote add origin $remote


        Write-Host "Creación exitosa!"
    }
    catch {
        Write-Host "Error al crear repositorio"
    }
}

function GenerarCommit {
    param (
        [string]$comentario
    )

    try {
        git init
        git add .
        git commit -m $comentario
        git branch -M main
        git push --set-upstream origin main

        Write-Host "Commit exitoso!"
    }
    catch {
        Write-Host "Error al crear commit"
    }
}

#------------------------------------------------------------

# cd E:\OMAR\OMARGR\MATERIALES_PROPIOS\PROYECTOS\API\API_speech_to_text

cd E:\OMAR\OMARGR\MATERIALES_PROPIOS\PROYECTOS\API\API_chat_pdf
<#
git remote rm <nombre>
git remote -v
rm -rf .git
Remove-Item '.git' -Recurse -Force
#>

$username ="Ogironr"
$name = "OmarGR"
$email = "omargiron940@gmail.com"
$nombre_repo = "API_speech_to_text"
CrearRepositorioGithub $username $name $email $nombre_repo

$comentario = "primer commit"
GenerarCommit $comentario

$comentario = "segundo commit"
GenerarCommit $comentario

$comentario = "tercer commit"
GenerarCommit $comentario

$comentario = "cuarto commit"
GenerarCommit $comentario

$comentario = "quito commit"
GenerarCommit $comentario

$comentario = "sexto commit"
GenerarCommit $comentario

$comentario = "7 commit"
GenerarCommit $comentario

$comentario = "8 commit"
GenerarCommit $comentario

#--------------------------------------------


$username ="Ogironr"
$name = "OmarGR"
$email = "omargiron940@gmail.com"
$nombre_repo = "API_chat_pdf"
CrearRepositorioGithub $username $name $email $nombre_repo

$comentario = "primer commit"
GenerarCommit $comentario

$comentario = "segundo commit"
GenerarCommit $comentario

$comentario = "tercer commit"
GenerarCommit $comentario