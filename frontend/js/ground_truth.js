(() => {
  const $ = id => document.getElementById(id);
  let selected = null, annotation = null, fieldStatuses = {};
  const msg = (text, error=false) => { $('message').textContent=text; $('message').classList.toggle('error',error); };
  async function api(url, options={}) {
    const response=await fetch(url,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});
    if(!response.ok){let detail=`Request failed (${response.status})`;try{detail=(await response.json()).detail||detail}catch{}throw new Error(detail)}
    return response.json();
  }
  function renderStatuses(){ $('statuses').replaceChildren(...Object.entries(fieldStatuses).map(([path,status])=>{const row=document.createElement('div');row.className='status-row';row.textContent=`${path}: ${status}`;return row;})); }
  function renderHistory(revisions=[]){$('revisions').replaceChildren(...revisions.map(r=>{const li=document.createElement('li');li.textContent=`#${r.revision} ${r.action} — ${new Date(r.created_at).toLocaleString()}`;return li;}));}
  async function loadDocuments(){
    try { const docs=await api('/ground-truth/documents'); const q=$('filter').value.toLowerCase(); $('documents').replaceChildren(...docs.filter(d=>`${d.filename} ${d.document_type}`.toLowerCase().includes(q)).map(d=>{const li=document.createElement('li');li.innerHTML='';const name=document.createElement('strong');name.textContent=d.filename||d.id;const meta=document.createElement('small');meta.textContent=`${d.document_type||'Document'} · ${d.annotation_status||'Not annotated'}`;li.append(name,meta);li.onclick=()=>selectDocument(d.id);return li;})); }
    catch(e){msg(e.message,true)}
  }
  async function selectDocument(id){
    try { const data=await api(`/ground-truth/documents/${encodeURIComponent(id)}`);selected=id;annotation=data.annotation;fieldStatuses=annotation?.field_statuses||{};
      $('doc-title').textContent=`${data.document.filename} — ${data.document.document_type||''}`;$('document-viewer').src=data.document.source_url;$('document-viewer').hidden=false;$('no-document').hidden=true;
      $('suggestion').textContent=data.suggestion?JSON.stringify(data.suggestion,null,2):'No extracted JSON available.';
      $('json-data').value=JSON.stringify(annotation?.data||{},null,2);$('workflow').value=annotation?.status||'Draft';renderStatuses();renderHistory(annotation?.revisions||[]);msg('');
    }catch(e){msg(e.message,true)}
  }
  async function save(status){
    if(!selected)throw new Error('Select a document first');let data;try{data=JSON.parse($('json-data').value)}catch(e){throw new Error(`Invalid JSON: ${e.message}`)}
    if(data===null||typeof data!=='object')throw new Error('Ground truth must be a JSON object or array');
    annotation=await api(`/ground-truth/documents/${encodeURIComponent(selected)}`,{method:'PUT',body:JSON.stringify({data,field_statuses:fieldStatuses,status})});renderHistory(annotation.revisions);$('workflow').value=annotation.status;await loadDocuments();return annotation;
  }
  $('filter').addEventListener('input',loadDocuments);
  $('set-status').onclick=()=>{const path=$('status-path').value.trim();if(!path){msg('Enter a JSON path for the field status.',true);return}fieldStatuses[path]=$('field-status').value;renderStatuses();$('status-path').value='';msg('Field status updated; save the annotation to persist it.')};
  $('save').onclick=async()=>{try{await save('Draft');msg('Draft saved. Model suggestions remain unverified.')}catch(e){msg(e.message,true)}};
  $('workflow-save').onclick=async()=>{try{await save($('workflow').value);const result=await api(`/ground-truth/documents/${encodeURIComponent(selected)}/workflow`,{method:'POST',body:JSON.stringify({status:$('workflow').value})});annotation=result;renderHistory(result.revisions);msg(`Status updated to ${result.status}.`);await loadDocuments()}catch(e){msg(e.message,true)}};
  $('verify').onclick=async()=>{try{await save('Draft');const confirmed=window.confirm('I inspected the original document and confirm this annotation as ground truth.');if(!confirmed)return;const result=await api(`/ground-truth/documents/${encodeURIComponent(selected)}/workflow`,{method:'POST',body:JSON.stringify({status:'Verified',confirmed:true})});annotation=result;renderHistory(result.revisions);$('workflow').value='Verified';msg('Annotation explicitly verified.');await loadDocuments()}catch(e){msg(e.message,true)}};
  $('delete').onclick=async()=>{if(!selected||!annotation)return msg('There is no saved annotation to delete.',true);if(!window.confirm('Delete this annotation and its revision history?'))return;try{await api(`/ground-truth/documents/${encodeURIComponent(selected)}`,{method:'DELETE'});annotation=null;fieldStatuses={};$('json-data').value='{}';renderStatuses();renderHistory();$('workflow').value='Draft';msg('Annotation deleted. The source document was preserved.');await loadDocuments()}catch(e){msg(e.message,true)}};
  loadDocuments();
})();
