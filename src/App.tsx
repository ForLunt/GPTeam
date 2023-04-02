import { useRef, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import 'semantic-ui-css/semantic.min.css'
import { Button, Header, Icon, Segment, Grid } from 'semantic-ui-react'
import Uploader from './Uploader'
import './App.css';

function App() {
  const uploadRef = useRef<HTMLInputElement>(null)

  return (
    <>
      <Grid>
        <Grid.Row>
          <Grid.Column width={2}/>
          <Grid.Column width={12} stretched textAlign="center" >
            <Uploader />
          </Grid.Column>
          <Grid.Column width={2}/>
        </Grid.Row>
        
      </Grid>
    </>
  )
}

export default App
