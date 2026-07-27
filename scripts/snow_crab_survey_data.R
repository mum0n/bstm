# create data from snow crab survey

  source( file.path( "~", ".Rprofile" )  )
  source( file.path( code_root, "bio_startup.R" )  )
   
  require(bio.snowcrab)   # loadfunctions("bio.snowcrab") 

  if (!exists("year.assessment" )) {
    if (exists( "yrs" )) {
      year.assessment = max(yrs) 
    } else {
      year.assessment = year(Sys.Date()) - 1
      yrs = 1999:year.assessment
    }
  }
  
  spec_bio = bio.taxonomy::taxonomy.recode( from="spec", to="parsimonious", tolookup=2526 )
  snowcrab_filter_class = "fb"     # fishable biomass (including soft-shelled )  "m.mat" "f.mat" "imm"
  
  carstm_model_label= paste( "default", snowcrab_filter_class, sep="_" )

  # params for number
  pN = snowcrab_parameters(
    project_class="carstm",
    yrs=yrs,   
    areal_units_type="tesselation",
    carstm_model_label= carstm_model_label,  
    selection = list(
      type = "number",
      biologicals=list( spec_bio=spec_bio ),
      biologicals_using_snowcrab_filter_class=snowcrab_filter_class
    )
  )

  if (0) {
      
    # params for mean size .. mostly the same as pN
    pW = snowcrab_parameters(
      project_class="carstm",
      yrs=yrs,   
      areal_units_type="tesselation",
      carstm_model_label= carstm_model_label,  
      selection = list(
        type = "meansize",
        biologicals=list( spec_bio=spec_bio ),
        biologicals_using_snowcrab_filter_class=snowcrab_filter_class
      )

    )

    # params for probability of observation
    pH = snowcrab_parameters( 
      project_class="carstm", 
      yrs=yrs,  
      areal_units_type="tesselation", 
      carstm_model_label= carstm_model_label,  
      selection = list(
        type = "presence_absence",
        biologicals=list( spec_bio=spec_bio ),
        biologicals_using_snowcrab_filter_class=snowcrab_filter_class
      )
    )


    redo_data = FALSE  # that is, re-create the base data "carstm_inputs"
    
    if (redo_data) {
      xydata = snowcrab.db( p=pN, DS="areal_units_input", redo=TRUE )
      sppoly = areal_units( p=pN, xydata=xydata, 
          spbuffer=5,  n_iter_drop=5, redo=TRUE, verbose=TRUE )  # create constrained polygons with neighbourhood as an attribute
      
      plot(sppoly["AUID"])

      sppoly$dummyvar = ""
      xydata = st_as_sf( xydata, coords=c("lon","lat") )
      st_crs(xydata) = st_crs( projection_proj4string("lonlat_wgs84") )

      additional_features = snowcrab_mapping_features(pN)  # for mapping below
    
      tmap_mode("plot")
      
      plt = 
        tm_shape(sppoly) +
          tm_borders(col = "slategray", alpha = 0.5, lwd = 0.5) + 
          tm_shape( xydata ) + tm_sf() +
          additional_features +
          tm_compass(position = c("right", "TOP"), size = 1.5) +
          tm_scale_bar(position = c("RIGHT", "BOTTOM"), width =0.1, text.size = 0.5) +
          tm_layout(frame = FALSE, scale = 2) +
          tm_shape( st_transform(polygons_rnaturalearth(), st_crs(sppoly) )) + 
          tm_borders(col = "slategray", alpha = 0.5, lwd = 0.5)

      dev.new(width=14, height=8, pointsize=20)
      plt
      
      M = snowcrab.db( p=pN, DS="carstm_inputs", sppoly=sppoly, redo=TRUE )  # will redo if not found
    }

  }
   
  sppoly = areal_units( p=pN )

  M = snowcrab.db( p=pN, DS="carstm_inputs", sppoly=sppoly  )  # will redo if not found
  M = M[ which(is.finite( M$t + M$z +M$pca1)), ]  # drop missings (72)

#   iq = unique( c( which( M$totno > 0), ip ) )
#   iw = unique( c( which( M$totno > 5), ip ) )  # need a good sample to estimate mean size
 
#   space_id = sppoly$AUID,
#   time_id =  pN$yrs,
#   cyclic_id = pN$cyclic_levels,


  # save location:
  if (!exists("outdir" )) {
    outdir = file.path( homedir, "projects", "model_covariance", "data", "snowcrab" ) 
  }

  dir.create(outdir, recursive=TRUE, showWarnings=FALSE )

  message("Saving R-data files to: ", outdir) 

  fn1 = file.path(outdir, "snowcrab_data.rdz" )
  read_write_fast( data=M, fn=fn1 )
  message("Saved: ", fn1) 
  
  fn2 = file.path(outdir, "snowcrab_nb.rdz" )
  nb = attributes(sppoly)$nb
  read_write_fast( data=nb, fn=fn2 )
  message("Saved: ", fn2) 
  
  fn3 = file.path(outdir, "snowcrab_sppoly.rdz" )
  sppoly = st_drop_geometry(sppoly)
  read_write_fast( data=sppoly, fn=fn3 )
  message("Saved: ", fn3) 

