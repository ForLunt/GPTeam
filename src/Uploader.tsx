import React, { useEffect, useRef, useState } from 'react'
import { Button, Header, Icon, Segment } from 'semantic-ui-react'

function Uploader() {
  const uploadRef = useRef<HTMLInputElement>(null)
  const [images, setImages] = useState<FileList |undefined>()
  
  useEffect(() => {
    if(images !== undefined) {
        console.log(images)
        const fdata = new FormData();

        for (let i = 0; i < images.length; i++) {
            const file = images[i];
            fdata.append(file.name, file);
        }
        fetch(`http://localhost/test`, {
            body : fdata,
            method : "POST",
        })
    }
    }, [images])
  return (
    <Segment placeholder >
    <Header icon>
      <Icon name='file image outline' />
      Ajouter des images
    </Header>
    <Button primary onClick={() => uploadRef.current.click()}>Uploader</Button>
    <input type="file" onChange={e => {if(e.target.files != undefined) setImages(e.target.files)}} ref={uploadRef} multiple style={{display : "none"}} accept=" image/jpeg"/>
  </Segment>
  )
}

export default Uploader;