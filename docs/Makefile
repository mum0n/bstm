FN?=FN_none
PD?=$(realpath $(dir $(abspath $(MAKEFILE_LIST)) )/../ )
WK?=$(PD)/ignore
YR?=2025
DOCTYPE?=html
PDFENGINE?=lualatex
TEMPLATE?=my_template.tex

# warning: file, dir, and year variables names can include terminal or starting spaces so strip them  ..
$FN=(strip $(FN))
$WK=(strip $(WK))


# export TEXINPUTS=.:media//:
# export BIBINPUTS=.:media//:
# export BSTINPUTS=.:media//:

# above are params that can be sent with make command (except PARAMS, below)
FNM := $(basename $(notdir $(FN)))
WKY := $(WK)/$(FNM)/$(YR)

PARAMS?="-P project_directory:$(PD) -P work_directory:$(WK) -P YR:$(YR) -P todo:[nil,nil]" # for quarto

# If the first argument is "run"...
ifeq (run, $(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif
 
# declare non-file targets:
.PHONY: dummy run all install clean cleanall distclean TAGS info check watch

.ONESHELL: copy # Applies to every target in the file!  .. to permit changing directory
 
test:
	@echo "$(PD)" 
	@echo "$(WKY)"
	@echo "$(PARAMS)"
 
dummy :
	@echo need to pass parameters to make
 
run : 
	@echo prog $(RUN_ARGS)  # just an example

all: latex pdf 

preparefiles : 
	@echo Copying files to a work area:  
	@echo $(WKY)
	mkdir -p $(WKY)
	cp -rf $(PD)/docs/media $(WKY)/
	cp -f $(PD)/docs/{Makefile,*.yml,*.bib,*.csl,*.qmd} $(WKY)/
	cp -f $(PD)/docs/$(FN) $(WKY)/

# quarto and rmarkdown fussy about file name extensions

quarto : preparefiles
	cd $(WKY)
	cp -f $(WKY)/$(FN) $(WKY)/$(FNM).qmd 	
	quarto render $(FNM).qmd --to $(DOCTYPE) $(PARAMS)
	ls $(WKY)/$(FNM).*
	
rmarkdown : preparefiles
	cd $(WKY)
	cp -f $(WKY)/$(FN) $(WKY)/$(FNM).rmd
	Rscript -e "rmarkdown::render('$(WKY)/$(FNM).rmd', params=list($(PARAMS)), output_format=$(DOCTYPE), output_dir=$(WKY))" 	
	ls $(WKY)/$(FNM).*
	
latex : preparefiles
	cd $(WKY)
	pandoc -s $(FN) \
	--from=markdown+tex_math_single_backslash+tex_math_dollars+raw_tex \
	--to=latex --template=$(TEMPLATE) \
	--output=$(FNM).tex 
	ls $(WKY)/$(FNM).*
	
	
latex2pdf : latex
	cd $(WKY)
	pdflatex $(FNM).tex
	biber $(FNM) 
	pdflatex $(FNM).tex
	pdflatex $(FNM).tex
	ls $(WKY)/$(FNM).*
	
	
pdf : preparefiles
	cd $(WKY)
	pandoc -s $(FN) \
	--from=markdown+tex_math_single_backslash+tex_math_dollars+raw_tex \
	--to=latex --template=$(TEMPLATE) \
	--output=$(FNM).pdf \
	--pdf-engine=$(PDFENGINE)
	pdflatex $(FNM).tex
	ls $(WKY)/$(FNM).*
	
	
html : preparefiles
	cd $(WKY)
	pandoc $(FN) \
	--from=markdown+tex_math_single_backslash+tex_math_dollars \
	--to=html5 \ 
	--output=$(FNM).html \
#	--mathjax \
	--embed-resources --standalone
	ls $(WKY)/$(FNM).*
	
	
epub : preparefiles
	cd $(WKY)
	pandoc $(FN) \
	--from=markdown+tex_math_single_backslash+tex_math_dollars \
	--to=epub \
	--output=$(FNM).epub \
	--epub-cover-image=<cover-image> \
	--toc
	ls $(WKY)/$(FNM).*
	
	
docx : preparefiles
	cd $(WKY)
	pandoc $(FN) \
	--from=markdown+tex_math_single_backslash+tex_math_dollars \
	--to=docx \
	--output=$(FNM).docx \
	ls $(WKY)/$(FNM).*
	
	
odt : preparefiles
	cd $(WKY)
	pandoc $(FN) \
	--from=markdown+tex_math_single_backslash+tex_math_dollars \
	--to=odt \
	--output=$(FNM).odt \
	ls $(WKY)/$(FNM).*
	
  
git:
	git commit -m"update"
	git checkout master
	git merge develop
	git push
	git pull
	git checkout develop
	git merge master
	git status
	

 
watch: $(FN).md 
	fswatch -o $^ | xargs -n1 -I{} make
# 	while true; do make -q || make; sleep 0.5; done


.PHONY: clean  
 
clean :
	cd $(WKY)
	rm -rf *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.spl  

cleanall: clean
	cd $(WKY)
	rm -rf *.docx *.odt *.epub *.html *.fls *.synctex.gz *.tex~ 
	rm -rf *.bbl-SAVE-ERROR *.bcf-SAVE-ERROR *.fdb_latexmk
	

